"""Run one privacy-bounded Codex benchmark over the local app-server protocol.

The only generated text is a fixed prompt and a short response. Protocol messages are
projected to IDs, item types, timestamps, terminal status, and token counts as soon as
they are read; text and unrecognized event payloads are discarded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TOOL_VERSION = "0.2.0"
_PROMPT_ID = "codex_speed_v3"
_BASE_INSTRUCTIONS = (
    "Single-turn speed benchmark. Answer the user's request directly. "
    "Do not call tools, use MCP, or delegate."
)
_PROMPT = (
    "Write about 700 words in English as a clear, vivid account of a small home garden "
    "changing through the four seasons. Follow one modest garden from the first signs of "
    "spring to winter rest. Describe plausible plants, weather, soil, light, insects, birds, "
    "and a few simple gardening tasks. Keep the details coherent and grounded, with a calm "
    "observational tone, varied sentences, and a natural beginning and ending. Avoid headings, "
    "lists, invented statistics, and a preamble. This is a single-turn speed benchmark: do not "
    "call tools, inspect files, use MCP, delegate, or ask questions."
)
_PROMPT_SHA256 = hashlib.sha256(f"{_BASE_INSTRUCTIONS}\n\n{_PROMPT}".encode()).hexdigest()
_MAX_EVENTS = 4096
_SAFE_LABEL = re.compile(r"[\w .:+@-]{1,100}\Z", re.UNICODE)
_SAFE_VERSION = re.compile(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?\Z")
_OUTPUT_ITEMS = {"reasoning", "agentMessage"}
_USER_ITEMS = {"userMessage"}
_EOF = object()
_INVALID = object()


class _BenchmarkTimeout(Exception):
    pass


class _ProtocolFailure(Exception):
    pass


@dataclass(slots=True)
class _Frame:
    kind: str
    received_at: float
    request_id: int | str | None = None
    method: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def _safe_label(value: object) -> str | None:
    if not isinstance(value, str) or not _SAFE_LABEL.fullmatch(value.strip()):
        return None
    value = value.strip()
    if re.fullmatch(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}", value):
        return None
    return value


def _nonnegative_int(value: object) -> int | None:
    if type(value) is int and 0 <= value <= 2**63 - 1:
        return value


def _benchmark_config(codex_home: Path) -> dict[str, Any]:
    """Disable configured MCPs and plugins only for this benchmark thread."""
    config: dict[str, Any] = {"mcp_servers": {}, "plugins": {}}
    try:
        with (codex_home / "config.toml").open("rb") as stream:
            user_config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        return config

    for section in ("mcp_servers", "plugins"):
        configured = user_config.get(section)
        if not isinstance(configured, dict):
            continue
        config[section] = {
            name: {"enabled": False}
            for name, value in configured.items()
            if isinstance(name, str) and isinstance(value, dict)
        }
    return config
    return None


def _timestamp_ms(value: object) -> float | None:
    """Normalize app-server Unix milliseconds to Unix seconds."""
    if type(value) is not int or value <= 0:
        return None
    return value / 1000.0


def _project_response(request_method: str, value: object) -> dict[str, Any]:
    """Keep only response fields needed for control flow and the benchmark record."""
    if not isinstance(value, dict):
        return {}
    if request_method == "initialize":
        server_info = value.get("serverInfo")
        version = server_info.get("version") if isinstance(server_info, dict) else None
        return {
            "codex_version": version
            if isinstance(version, str) and _SAFE_VERSION.fullmatch(version)
            else None
        }
    if request_method == "thread/start":
        thread = value.get("thread")
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        sandbox = value.get("sandbox")
        return {
            "thread_id": thread_id if isinstance(thread_id, str) else None,
            "model": _safe_label(value.get("model")),
            "effort": _safe_label(value.get("reasoningEffort")),
            "service_tier": _safe_label(value.get("serviceTier")),
            "approval_never": value.get("approvalPolicy") == "never",
            "sandbox_read_only": isinstance(sandbox, dict)
            and sandbox.get("type") == "readOnly"
            and sandbox.get("networkAccess", False) is False,
        }
    if request_method == "turn/start":
        turn = value.get("turn")
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        status = turn.get("status") if isinstance(turn, dict) else None
        return {
            "turn_id": turn_id if isinstance(turn_id, str) else None,
            "turn_status": status if isinstance(status, str) else None,
        }
    return {}


def _project_notification(method: str, params: object, received_at: float) -> _Frame | None:
    if not isinstance(params, dict):
        return None
    if method in {"item/started", "item/completed"}:
        item = params.get("item")
        if not isinstance(item, dict):
            return None
        item_id = item.get("id")
        item_type = item.get("type")
        if not isinstance(item_id, str) or not isinstance(item_type, str):
            return None
        timestamp_key = "startedAtMs" if method == "item/started" else "completedAtMs"
        timestamp = _timestamp_ms(params.get(timestamp_key))
        return _Frame(
            "notification",
            received_at,
            method=method,
            payload={
                "thread_id": params.get("threadId"),
                "turn_id": params.get("turnId"),
                "item_id": item_id,
                "item_type": item_type,
                "timestamp": timestamp,
            },
        )
    if method == "thread/tokenUsage/updated":
        usage = params.get("tokenUsage")
        last = usage.get("last") if isinstance(usage, dict) else None
        if not isinstance(last, dict):
            return None
        return _Frame(
            "notification",
            received_at,
            method=method,
            payload={
                "thread_id": params.get("threadId"),
                "turn_id": params.get("turnId"),
                "usage": {
                    name: _nonnegative_int(last.get(source_name))
                    for name, source_name in (
                        ("output_tokens", "outputTokens"),
                        ("input_tokens", "inputTokens"),
                        ("cached_input_tokens", "cachedInputTokens"),
                        ("reasoning_output_tokens", "reasoningOutputTokens"),
                    )
                },
            },
        )
    if method == "turn/completed":
        turn = params.get("turn")
        if not isinstance(turn, dict):
            return None
        return _Frame(
            "notification",
            received_at,
            method=method,
            payload={
                "thread_id": params.get("threadId"),
                "turn_id": turn.get("id"),
                "turn_status": turn.get("status") if isinstance(turn.get("status"), str) else None,
            },
        )
    return None


class _ProtocolClient:
    """Minimal JSON-RPC stdio client that projects and discards incoming payloads."""

    def __init__(self, process: subprocess.Popen[str]):
        self.process = process
        self.frames: queue.Queue[_Frame | object] = queue.Queue()
        self.request_sequence = 0
        self.pending: dict[int, str] = {}
        self.pending_lock = threading.Lock()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            self.frames.put(_EOF)
            return
        while True:
            try:
                line = stream.readline()
            except Exception:
                self.frames.put(_INVALID)
                return
            if not line:
                self.frames.put(_EOF)
                return
            received_at = time.monotonic()
            try:
                message = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                self.frames.put(_INVALID)
                return
            if not isinstance(message, dict):
                self.frames.put(_INVALID)
                return

            if "method" in message and "id" in message:
                request_id = message.get("id")
                if isinstance(request_id, (int, str)) and not isinstance(request_id, bool):
                    self.frames.put(_Frame("server_request", received_at, request_id=request_id))
                continue

            if "id" in message:
                request_id = message.get("id")
                if isinstance(request_id, (int, str)) and not isinstance(request_id, bool):
                    with self.pending_lock:
                        request_method = self.pending.pop(request_id, None)
                    if request_method is not None:
                        result = message.get("result")
                        payload = _project_response(request_method, result)
                        self.frames.put(
                            _Frame(
                                "response",
                                received_at,
                                request_id=request_id,
                                method=request_method,
                                payload={"error": "error" in message, **payload},
                            )
                        )
                continue

            method = message.get("method")
            if isinstance(method, str):
                frame = _project_notification(method, message.get("params"), received_at)
                if frame is not None:
                    self.frames.put(frame)

    def _write(self, message: dict[str, Any]) -> None:
        stream = self.process.stdin
        if stream is None or self.process.poll() is not None:
            raise _ProtocolFailure
        stream.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _next(self, deadline: float) -> _Frame:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _BenchmarkTimeout
        try:
            frame = self.frames.get(timeout=remaining)
        except queue.Empty as exc:
            raise _BenchmarkTimeout from exc
        if frame is _EOF:
            raise _ProtocolFailure
        if frame is _INVALID or not isinstance(frame, _Frame):
            raise _ProtocolFailure
        if frame.kind == "server_request":
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": frame.request_id,
                    "error": {"code": -32601, "message": "Unsupported request"},
                }
            )
            return self._next(deadline)
        return frame

    def request(
        self,
        method: str,
        params: dict[str, Any],
        deadline: float,
        collector: _EventCollector | None = None,
    ) -> _Frame:
        self.request_sequence += 1
        request_id = self.request_sequence
        with self.pending_lock:
            self.pending[request_id] = method
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            frame = self._next(deadline)
            if frame.kind == "response" and frame.request_id == request_id:
                return frame
            if frame.kind == "notification" and collector is not None:
                collector.add(frame)

    def next_notification(self, deadline: float, collector: _EventCollector) -> _Frame:
        while True:
            frame = self._next(deadline)
            if frame.kind == "notification":
                collector.add(frame)
                return frame


@dataclass(slots=True)
class _ItemTiming:
    item_type: str
    started: float | None = None
    completed: float | None = None
    first_seen: float | None = None


class _EventCollector:
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.events: list[_Frame] = []
        self.overflowed = False

    def add(self, frame: _Frame) -> None:
        payload = frame.payload
        if payload.get("thread_id") != self.thread_id:
            return
        if len(self.events) >= _MAX_EVENTS:
            self.overflowed = True
            return
        self.events.append(frame)

    def _for_turn(self, turn_id: str) -> list[_Frame]:
        return [frame for frame in self.events if frame.payload.get("turn_id") == turn_id]

    def completed(self, turn_id: str) -> _Frame | None:
        return next(
            (
                frame
                for frame in reversed(self._for_turn(turn_id))
                if frame.method == "turn/completed"
            ),
            None,
        )

    def has_usage(self, turn_id: str) -> bool:
        return any(frame.method == "thread/tokenUsage/updated" for frame in self._for_turn(turn_id))

    def has_tool_item(self, turn_id: str) -> bool:
        return any(
            frame.method in {"item/started", "item/completed"}
            and frame.payload.get("item_type") not in _OUTPUT_ITEMS | _USER_ITEMS
            for frame in self._for_turn(turn_id)
        )

    def measurement(
        self, turn_id: str, turn_sent_at: float
    ) -> tuple[dict[str, int | None], float | None, float | None, list[str], float | None]:
        frames = self._for_turn(turn_id)
        diagnostics: list[str] = []
        items: dict[str, _ItemTiming] = {}
        usage: dict[str, int | None] | None = None
        turn_done: _Frame | None = None
        for frame in frames:
            if frame.method == "thread/tokenUsage/updated":
                usage = frame.payload.get("usage")
            elif frame.method == "turn/completed":
                turn_done = frame
            elif frame.method in {"item/started", "item/completed"}:
                item_id = frame.payload.get("item_id")
                item_type = frame.payload.get("item_type")
                if not isinstance(item_id, str) or not isinstance(item_type, str):
                    diagnostics.append("invalid_item_event")
                    continue
                if item_type in _USER_ITEMS:
                    continue
                if item_type not in _OUTPUT_ITEMS:
                    diagnostics.append("tool_used")
                    continue
                entry = items.setdefault(item_id, _ItemTiming(item_type))
                if entry.item_type != item_type:
                    diagnostics.append("item_mismatch")
                    continue
                timestamp = frame.payload.get("timestamp")
                if timestamp is None:
                    diagnostics.append("missing_item_timing")
                    continue
                if frame.method == "item/started":
                    if entry.started is not None:
                        diagnostics.append("duplicate_item_start")
                    entry.started = timestamp
                    entry.first_seen = frame.received_at
                else:
                    if entry.completed is not None:
                        diagnostics.append("duplicate_item_completion")
                    entry.completed = timestamp

        output_items = list(items.values())
        if not output_items:
            diagnostics.append("missing_output_item")
        if not any(item.item_type == "agentMessage" for item in output_items):
            diagnostics.append("missing_agent_message")
        if any(item.started is None for item in output_items):
            diagnostics.append("missing_item_start")
        if any(item.completed is None for item in output_items):
            diagnostics.append("missing_item_completion")
        if usage is None:
            diagnostics.append("missing_usage")
            usage = {}
        required_usage = (
            "output_tokens",
            "input_tokens",
            "cached_input_tokens",
            "reasoning_output_tokens",
        )
        if any(usage.get(name) is None for name in required_usage):
            diagnostics.append("invalid_usage")
        reasoning_tokens = usage.get("reasoning_output_tokens")
        has_reasoning_item = any(item.item_type == "reasoning" for item in output_items)
        if type(reasoning_tokens) is int and reasoning_tokens > 0 and not has_reasoning_item:
            diagnostics.append("missing_reasoning_timing")
        if self.overflowed:
            diagnostics.append("event_limit_exceeded")
        if turn_done is None:
            diagnostics.append("missing_turn_completion")
        elif turn_done.payload.get("turn_status") != "completed":
            diagnostics.append("turn_failed")

        starts = [item.started for item in output_items if item.started is not None]
        completions = [item.completed for item in output_items if item.completed is not None]
        if any(
            item.started is not None
            and item.completed is not None
            and item.completed < item.started
            for item in output_items
        ):
            diagnostics.append("invalid_item_timing")
        stream_seconds: float | None = None
        first_output_seconds: float | None = None
        end_to_end_seconds: float | None = None
        if starts and completions:
            start = min(starts)
            end = max(completions)
            if end > start:
                stream_seconds = end - start
            else:
                diagnostics.append("invalid_item_timing")
        first_seen = [item.first_seen for item in output_items if item.first_seen is not None]
        if first_seen:
            first_output_seconds = max(0.0, min(first_seen) - turn_sent_at)
        if turn_done is not None:
            end_to_end_seconds = max(0.0, turn_done.received_at - turn_sent_at)
        if stream_seconds is None:
            diagnostics.append("missing_stream_timing")

        # Keep diagnostic order stable while avoiding duplicate reason codes.
        diagnostics = list(dict.fromkeys(diagnostics))
        return usage, stream_seconds, first_output_seconds, diagnostics, end_to_end_seconds


def _base_result(
    *,
    status: str,
    model: str | None,
    effort: str | None,
    started_at: float,
    diagnostics: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "requested_model": _safe_label(model),
        "observed_model": None,
        "effort": _safe_label(effort),
        "provider": None,
        "service_tier": None,
        "prompt_id": _PROMPT_ID,
        "prompt_sha256": _PROMPT_SHA256,
        "tool_version": _TOOL_VERSION,
        "codex_version": None,
        "output_tokens": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "reasoning_output_tokens": None,
        "stream_seconds": None,
        "end_to_end_seconds": None,
        "first_output_seconds": None,
        "tps": None,
        "end_to_end_tps": None,
        "measurement_method": None,
        "valid_samples": 0,
        "excluded_samples": 1 if diagnostics else 0,
        "exclusions": {},
        "diagnostics": list(dict.fromkeys(diagnostics)),
        "samples": [],
        "started_at": started_at,
        "ended_at": time.time(),
    }


def _stop_process(process: subprocess.Popen[str]) -> None:
    stream = process.stdin
    if stream is not None:
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    try:
        process.wait(timeout=0.5)
        return
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        process.terminate()
        process.wait(timeout=1.0)
        return
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        process.kill()
        process.wait(timeout=1.0)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _cleanup_scratch(scratch: tempfile.TemporaryDirectory[str]) -> None:
    try:
        scratch.cleanup()
    except OSError:
        # Keep the result record usable if the host denies temporary-directory cleanup.
        # The Codex thread itself is still ephemeral and the process is already stopped.
        pass


def run(
    model: str | None = None,
    effort: str | None = None,
    timeout: float = 600,
    source: Path | None = None,
    codex_command: str | Path | None = None,
) -> dict[str, Any]:
    """Run one no-tools ephemeral turn and return its measured speed record.

    ``source`` is the Codex home directory whose existing authorization should be reused.
    ``codex_command`` optionally names the Codex executable. The runner never persists data.
    """
    started_at = time.time()
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        return _base_result(
            status="failed",
            model=model,
            effort=effort,
            started_at=started_at,
            diagnostics=["invalid_timeout"],
        )

    deadline = time.monotonic() + float(timeout)
    executable = str(codex_command) if codex_command is not None else "codex"
    process: subprocess.Popen[str] | None = None
    client: _ProtocolClient | None = None
    observed_model: str | None = None
    observed_effort: str | None = None
    service_tier: str | None = None
    codex_version: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    turn_sent_at: float | None = None
    collector: _EventCollector | None = None

    scratch: tempfile.TemporaryDirectory[str] | None = None
    try:
        # An empty cwd avoids loading repository AGENTS.md and skills into the test turn.
        scratch = tempfile.TemporaryDirectory(prefix="codex-speed-")
        scratch_cwd = scratch.name
        try:
            env = os.environ.copy()
            if source is not None:
                env["CODEX_HOME"] = str(Path(source).expanduser().resolve())
            codex_home = Path(env.get("CODEX_HOME") or (Path.home() / ".codex"))
            thread_config = _benchmark_config(codex_home)
            process = subprocess.Popen(
                [executable, "app-server", "--listen", "stdio://"],
                cwd=scratch_cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            client = _ProtocolClient(process)

            initialized = client.request(
                "initialize",
                {
                    "clientInfo": {"name": "token-pulse", "version": "0.2.0"},
                    "capabilities": {"experimentalApi": True},
                },
                deadline,
            )
            if initialized.payload.get("error"):
                return _base_result(
                    status="failed",
                    model=model,
                    effort=effort,
                    started_at=started_at,
                    diagnostics=["initialize_failed"],
                )
            codex_version = initialized.payload.get("codex_version")
            client.notify("initialized")

            thread_params: dict[str, Any] = {
                "ephemeral": True,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "cwd": scratch_cwd,
                "config": thread_config,
                "dynamicTools": [],
                "environments": [],
                "multiAgentMode": "explicitRequestOnly",
                "baseInstructions": _BASE_INSTRUCTIONS,
            }
            if effort is not None:
                thread_params["config"]["model_reasoning_effort"] = effort
            if model is not None:
                thread_params["model"] = model
            thread_start = client.request("thread/start", thread_params, deadline)
            if thread_start.payload.get("error"):
                return _base_result(
                    status="failed",
                    model=model,
                    effort=effort,
                    started_at=started_at,
                    diagnostics=["thread_start_failed"],
                )
            thread_id = thread_start.payload.get("thread_id")
            observed_model = thread_start.payload.get("model")
            observed_effort = thread_start.payload.get("effort")
            service_tier = thread_start.payload.get("service_tier")
            if not isinstance(thread_id, str) or not thread_id:
                return _base_result(
                    status="failed",
                    model=model,
                    effort=effort,
                    started_at=started_at,
                    diagnostics=["missing_thread_id"],
                )
            if not thread_start.payload.get("approval_never") or not thread_start.payload.get(
                "sandbox_read_only"
            ):
                result = _base_result(
                    status="failed",
                    model=model,
                    effort=effort,
                    started_at=started_at,
                    diagnostics=["unsafe_policy"],
                )
                result["observed_model"] = observed_model
                result["effort"] = observed_effort or _safe_label(effort)
                result["service_tier"] = service_tier
                return result

            collector = _EventCollector(thread_id)
            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": [{"type": "text", "text": _PROMPT}],
            }
            if effort is not None:
                turn_params["effort"] = effort
            if model is not None:
                turn_params["model"] = model
            turn_sent_at = time.monotonic()
            turn_start = client.request("turn/start", turn_params, deadline, collector)
            if turn_start.payload.get("error"):
                result = _base_result(
                    status="failed",
                    model=model,
                    effort=effort,
                    started_at=started_at,
                    diagnostics=["turn_start_failed"],
                )
                result["observed_model"] = observed_model
                result["effort"] = observed_effort or _safe_label(effort)
                result["service_tier"] = service_tier
                return result
            turn_id = turn_start.payload.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                result = _base_result(
                    status="incomplete",
                    model=model,
                    effort=effort,
                    started_at=started_at,
                    diagnostics=["missing_turn_id"],
                )
                result["observed_model"] = observed_model
                result["effort"] = observed_effort or _safe_label(effort)
                result["service_tier"] = service_tier
                return result

            cancel_sent = False
            while collector.completed(turn_id) is None:
                if collector.has_tool_item(turn_id) and not cancel_sent:
                    cancel_sent = True
                    try:
                        client.request(
                            "turn/interrupt",
                            {"threadId": thread_id, "turnId": turn_id},
                            deadline,
                            collector,
                        )
                    except (_BenchmarkTimeout, _ProtocolFailure):
                        raise
                    except Exception:
                        pass
                    continue
                client.next_notification(deadline, collector)

            # Usage may follow turn/completed by a separate notification. Give that
            # same-turn update a short bounded grace period, without extending timeout.
            usage_deadline = min(deadline, time.monotonic() + 0.75)
            # Always listen briefly after turn completion so a final usage update can
            # supersede any earlier in-turn estimate.
            while time.monotonic() < usage_deadline:
                try:
                    client.next_notification(usage_deadline, collector)
                except _BenchmarkTimeout:
                    break
                except _ProtocolFailure:
                    break

            usage, stream_seconds, first_output_seconds, diagnostics, end_to_end_seconds = (
                collector.measurement(turn_id, turn_sent_at)
            )
            completed_frame = collector.completed(turn_id)
            if completed_frame is not None:
                status_value = completed_frame.payload.get("turn_status")
                if status_value in {"interrupted", "cancelled", "canceled"}:
                    status = "interrupted"
                    diagnostics.append("turn_interrupted")
                elif status_value != "completed":
                    status = "failed"
                else:
                    status = "completed"
            else:
                status = "incomplete"
            if collector.has_tool_item(turn_id):
                status = "incomplete"
                diagnostics.append("tool_used")
            if model is not None:
                if observed_model is None:
                    status = "incomplete"
                    diagnostics.append("model_unverified")
                elif model != observed_model:
                    status = "incomplete"
                    diagnostics.append("model_rerouted")
            if effort is not None:
                if observed_effort is None:
                    status = "incomplete"
                    diagnostics.append("effort_unverified")
                elif effort != observed_effort:
                    status = "incomplete"
                    diagnostics.append("effort_rerouted")
            diagnostics = list(dict.fromkeys(diagnostics))

            result = _base_result(
                status=status,
                model=model,
                effort=effort,
                started_at=started_at,
                diagnostics=diagnostics,
            )
            result.update(
                {
                    "observed_model": observed_model,
                    "codex_version": codex_version,
                    "effort": observed_effort or _safe_label(effort),
                    "service_tier": service_tier,
                    "output_tokens": usage.get("output_tokens"),
                    "input_tokens": usage.get("input_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                    "stream_seconds": stream_seconds,
                    "end_to_end_seconds": end_to_end_seconds,
                    "first_output_seconds": first_output_seconds,
                }
            )
            output_tokens = usage.get("output_tokens")
            if output_tokens == 0:
                result["status"] = "incomplete"
                result["diagnostics"] = list(dict.fromkeys([*diagnostics, "zero_output_tokens"]))
            if (
                result["status"] == "completed"
                and type(output_tokens) is int
                and output_tokens > 0
                and stream_seconds is not None
                and stream_seconds > 0
                and not diagnostics
            ):
                result["tps"] = output_tokens / stream_seconds
                result["measurement_method"] = "app_server_stream_estimate"
                result["valid_samples"] = 1
                result["excluded_samples"] = 0
                if end_to_end_seconds and end_to_end_seconds > 0:
                    result["end_to_end_tps"] = output_tokens / end_to_end_seconds
            else:
                if result["status"] == "completed":
                    result["status"] = "incomplete"
                result["measurement_method"] = None
                result["valid_samples"] = 0
                result["excluded_samples"] = 1
            result["diagnostics"] = list(dict.fromkeys(result["diagnostics"]))
            result["ended_at"] = time.time()
            return result
        finally:
            if process is not None:
                _stop_process(process)
            _cleanup_scratch(scratch)
            scratch = None

    except _BenchmarkTimeout:
        result = _base_result(
            status="timeout",
            model=model,
            effort=effort,
            started_at=started_at,
            diagnostics=["timeout"],
        )
        result["observed_model"] = observed_model
        result["effort"] = observed_effort or _safe_label(effort)
        result["service_tier"] = service_tier
        return result
    except KeyboardInterrupt:
        result = _base_result(
            status="interrupted",
            model=model,
            effort=effort,
            started_at=started_at,
            diagnostics=["interrupted"],
        )
        result["observed_model"] = observed_model
        result["effort"] = observed_effort or _safe_label(effort)
        result["service_tier"] = service_tier
        return result
    except FileNotFoundError:
        return _base_result(
            status="failed",
            model=model,
            effort=effort,
            started_at=started_at,
            diagnostics=["appserver_unavailable"],
        )
    except (_ProtocolFailure, OSError, ValueError, TypeError):
        result = _base_result(
            status="failed",
            model=model,
            effort=effort,
            started_at=started_at,
            diagnostics=["protocol_error"],
        )
        result["observed_model"] = observed_model
        result["effort"] = observed_effort or _safe_label(effort)
        result["service_tier"] = service_tier
        return result
    except Exception:
        result = _base_result(
            status="failed",
            model=model,
            effort=effort,
            started_at=started_at,
            diagnostics=["appserver_error"],
        )
        result["observed_model"] = observed_model
        result["effort"] = observed_effort or _safe_label(effort)
        result["service_tier"] = service_tier
        return result
    finally:
        # This outer cleanup also covers errors raised before the inner process scope.
        if process is not None and process.poll() is None:
            _stop_process(process)
        if scratch is not None:
            _cleanup_scratch(scratch)
