"""Extract only scoped tier settings from bounded Rust Debug submission records."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SubmittedTier:
    at: float
    tier: str | None
    state: str  # set / cleared / unspecified / ambiguous


def _fields(value: str, name: str) -> dict[str, str]:
    prefix = name + " {"
    if not value.startswith(prefix) or not value.endswith("}"):
        raise ValueError("Unexpected Debug structure")
    content = value[len(prefix) : -1]
    result = {}
    stack = []
    quoted = escaped = False
    start = 0
    for i, char in enumerate(content + ","):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "{[(":
            stack.append(char)
            if len(stack) > 64:
                raise ValueError("Debug nesting too deep")
        elif char in "}])":
            if not stack or stack.pop() != {"}": "{", "]": "[", ")": "("}[char]:
                raise ValueError("Unbalanced Debug structure")
        elif char == "," and not stack:
            part = content[start:i].strip()
            start = i + 1
            if not part:
                continue
            key, colon, raw = part.partition(":")
            if not colon or not re.fullmatch(r"[a-z_]+", key) or key in result:
                raise ValueError("Invalid or duplicate field")
            result[key] = raw.strip()
    if stack or quoted:
        raise ValueError("Truncated Debug structure")
    return result


def submitted_tier(task: str, at: float, body: str) -> tuple[str, SubmittedTier] | None:
    if len(body) > 16000:
        return None
    header = re.fullmatch(
        r'session_loop\{thread_id=(?:"([\w-]{1,100})"|([\w-]{1,100}))\}: '
        r"Submission sub=(.*)",
        body,
        re.DOTALL,
    )
    if not header or (header[1] or header[2]) != task:
        return None
    try:
        submission = _fields(header[3], "Submission")
        identifier = re.fullmatch(r'"([\w-]{1,100})"', submission.get("id", ""))
        op = _fields(submission.get("op", ""), "TurnInput")
        if not identifier or op.get("mode") not in {"Start", "StartOrSteer"}:
            return None
        request = _fields(op.get("request", ""), "TurnInputRequest")
        settings = _fields(request.get("thread_settings", ""), "ThreadSettingsOverrides")
        start = _fields(request.get("start", ""), "TurnStartOptions")
        raw = settings.get("service_tier")
        match = re.fullmatch(r'Some\(Some\("([a-z][a-z0-9_-]{0,39})"\)\)', raw or "")
        if match:
            tier, state = match[1], "set"
        elif raw in {"Some(None)", "None"}:
            tier, state = None, "cleared" if raw == "Some(None)" else "unspecified"
        else:
            return None
        # A second override position has not been verified for this adapter. Do
        # not guess its precedence, even if the thread settings contain a value.
        if start.get("service_tier", "None") != "None":
            tier, state = None, "ambiguous"
        return identifier[1], SubmittedTier(at, tier, state)
    except ValueError:
        return None
