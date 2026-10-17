from __future__ import annotations

import json
import queue
import subprocess
from pathlib import Path
from typing import Any

import pytest

from token_pulse import benchmark

_THREAD_ID = "thread-private-id"
_TURN_ID = "turn-private-id"
_OUTPUT = "private benchmark response text"
_BASE_MS = 1_760_000_000_000


@pytest.fixture(autouse=True)
def _use_workspace_tempdir(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(benchmark.tempfile, "tempdir", str(tmp_path))


class _FakeStdout:
    def __init__(self) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()

    def readline(self) -> str:
        line = self.lines.get()
        return "" if line is None else line

    def put(self, value: dict[str, Any]) -> None:
        self.lines.put(json.dumps(value) + "\n")

    def close(self) -> None:
        self.lines.put(None)


class _FakeStdin:
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process
        self.closed = False

    def write(self, line: str) -> int:
        message = json.loads(line)
        self.process.requests.append(message)
        self.process.handle(message)
        return len(line)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.process.returncode = 0
            self.process.stdout.close()


class _FakeProcess:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.stdout = _FakeStdout()
        self.requests: list[dict[str, Any]] = []
        self.stdin = _FakeStdin(self)
        self.returncode: int | None = None
        self.started = False

    def handle(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if method == "initialize":
            if self.scenario == "rpc_error":
                self.stdout.put(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {"code": -32000, "message": "private server error"},
                    }
                )
            else:
                self._respond(
                    message,
                    {"serverInfo": {"name": "codex", "version": "0.123.4"}},
                )
        elif method == "thread/start":
            params = message["params"]
            config = params.get("config", {})
            self._respond(
                message,
                {
                    "thread": {"id": _THREAD_ID},
                    "model": params.get("model", "gpt-6.1-sol"),
                    "reasoningEffort": config.get("model_reasoning_effort", "ultra"),
                    "serviceTier": "default",
                    "approvalPolicy": "never",
                    "sandbox": {"type": "readOnly", "networkAccess": False},
                },
            )
        elif method == "turn/start":
            if self.scenario == "timeout":
                return
            if self.scenario == "tool":
                self._tool_events()
            else:
                self._output_events(
                    include_usage=self.scenario != "missing_usage",
                    include_complete=self.scenario != "missing_completion",
                    include_reasoning=self.scenario in {"success", "reasoning"},
                    missing_timing=self.scenario == "missing_timing",
                    reverse_item=self.scenario == "reverse_item",
                    reasoning_tokens=0 if self.scenario == "reverse_item" else 25,
                )
            self._respond(message, {"turn": {"id": _TURN_ID, "status": "inProgress"}})
            if self.scenario != "tool":
                self._turn_completed()
        elif method == "turn/interrupt":
            self._respond(message, {})
            self._turn_completed("interrupted")

    def _respond(self, request: dict[str, Any], result: dict[str, Any]) -> None:
        self.stdout.put({"jsonrpc": "2.0", "id": request["id"], "result": result})

    def _notification(self, method: str, params: dict[str, Any]) -> None:
        self.stdout.put({"jsonrpc": "2.0", "method": method, "params": params})

    def _item(self, item_id: str, item_type: str, time_ms: int, *, completed: bool) -> None:
        item = {"id": item_id, "type": item_type, "text": _OUTPUT}
        name = "item/completed" if completed else "item/started"
        time_name = "completedAtMs" if completed else "startedAtMs"
        self._notification(
            name,
            {
                "threadId": _THREAD_ID,
                "turnId": _TURN_ID,
                "item": item,
                time_name: time_ms,
            },
        )

    def _output_events(
        self,
        *,
        include_usage: bool,
        include_complete: bool,
        include_reasoning: bool,
        missing_timing: bool = False,
        reverse_item: bool = False,
        reasoning_tokens: int = 25,
    ) -> None:
        if include_reasoning:
            self._item("reasoning-private-id", "reasoning", _BASE_MS, completed=False)
            self._item("reasoning-private-id", "reasoning", _BASE_MS + 3_000, completed=True)
        self._item(
            "message-private-id",
            "agentMessage",
            _BASE_MS + (5_000 if reverse_item else 1_000),
            completed=False,
        )
        if include_complete:
            self._item(
                "message-private-id",
                "agentMessage",
                _BASE_MS + (4_000 if reverse_item else 5_000),
                completed=True,
            )
        if missing_timing:
            self.stdout.put(
                {
                    "jsonrpc": "2.0",
                    "method": "item/started",
                    "params": {
                        "threadId": _THREAD_ID,
                        "turnId": _TURN_ID,
                        "item": {
                            "id": "untimed-message-private-id",
                            "type": "agentMessage",
                            "text": _OUTPUT,
                        },
                    },
                }
            )
        if include_usage:
            self._notification(
                "thread/tokenUsage/updated",
                {
                    "threadId": _THREAD_ID,
                    "turnId": _TURN_ID,
                    "tokenUsage": {
                        "last": {
                            "outputTokens": 100,
                            "inputTokens": 20,
                            "cachedInputTokens": 7,
                            "reasoningOutputTokens": reasoning_tokens,
                        }
                    },
                },
            )

    def _tool_events(self) -> None:
        self._notification(
            "item/started",
            {
                "threadId": _THREAD_ID,
                "turnId": _TURN_ID,
                "item": {"id": "tool-private-id", "type": "commandExecution"},
                "startedAtMs": _BASE_MS,
            },
        )

    def _turn_completed(self, status: str = "completed") -> None:
        self._notification(
            "turn/completed",
            {
                "threadId": _THREAD_ID,
                "turn": {"id": _TURN_ID, "status": status, "items": []},
            },
        )

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("codex", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self.stdout.close()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()


def _install_fake(monkeypatch: Any, scenario: str) -> tuple[_FakeProcess, dict[str, Any]]:
    process = _FakeProcess(scenario)
    captured: dict[str, Any] = {}

    def popen(args: list[str], **kwargs: Any) -> _FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["cwd_existed_at_launch"] = Path(kwargs["cwd"]).is_dir()
        return process

    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    return process, captured


def test_run_uses_ephemeral_read_only_single_turn_and_server_item_span(
    monkeypatch: Any, tmp_path: Path
) -> None:
    process, captured = _install_fake(monkeypatch, "success")
    source = tmp_path / "codex-home"
    source.mkdir()

    result = benchmark.run(
        model="gpt-6-luna", effort="low", timeout=2, source=source, codex_command="codex-test"
    )

    assert result["status"] == "completed"
    assert result["output_tokens"] == 100
    assert result["reasoning_output_tokens"] == 25
    assert result["tool_version"] == "0.2.0"
    assert result["codex_version"] == "0.123.4"
    # The union spans reasoning start through the final agent-message completion.
    assert result["stream_seconds"] == 5
    assert result["tps"] == 20
    assert result["measurement_method"] == "app_server_stream_estimate"
    assert result["diagnostics"] == []
    assert result["valid_samples"] == 1
    assert result["requested_model"] == result["observed_model"] == "gpt-6-luna"
    assert result["effort"] == "low"
    assert captured["args"] == ["codex-test", "app-server", "--listen", "stdio://"]
    assert captured["kwargs"]["env"]["CODEX_HOME"] == str(source)
    assert captured["cwd_existed_at_launch"] is True
    assert not Path(captured["kwargs"]["cwd"]).exists()
    thread_start = next(row for row in process.requests if row.get("method") == "thread/start")
    assert thread_start["params"]["ephemeral"] is True
    assert thread_start["params"]["approvalPolicy"] == "never"
    assert thread_start["params"]["sandbox"] == "read-only"
    assert thread_start["params"]["config"] == {
        "mcp_servers": {},
        "plugins": {},
        "model_reasoning_effort": "low",
    }
    assert thread_start["params"]["baseInstructions"] == benchmark._BASE_INSTRUCTIONS
    assert thread_start["params"]["dynamicTools"] == []
    assert thread_start["params"]["environments"] == []
    turn_start = next(row for row in process.requests if row.get("method") == "turn/start")
    assert turn_start["params"]["effort"] == "low"
    initialize = next(row for row in process.requests if row.get("method") == "initialize")
    assert initialize["params"]["capabilities"]["experimentalApi"] is True
    assert _OUTPUT not in repr(result)
    assert _THREAD_ID not in repr(result)
    assert _TURN_ID not in repr(result)
    assert result["prompt_id"] == "codex_speed_v3"
    assert (
        result["prompt_sha256"]
        == benchmark.hashlib.sha256(
            f"{benchmark._BASE_INSTRUCTIONS}\n\n{benchmark._PROMPT}".encode()
        ).hexdigest()
    )
    assert list(source.iterdir()) == []
    assert process.poll() == 0


def test_per_thread_disables_configured_mcp_servers_and_plugins(
    monkeypatch: Any, tmp_path: Path
) -> None:
    process, _ = _install_fake(monkeypatch, "success")
    source = tmp_path / "codex-home"
    source.mkdir()
    (source / "config.toml").write_text(
        '[mcp_servers."docs"]\ncommand = "ignored"\n'
        '[plugins."repo-tools@local"]\nenabled = true\n'
        '[plugins."already-off@local"]\nenabled = false\n',
        encoding="utf-8",
    )

    result = benchmark.run(timeout=2, source=source)

    assert result["status"] == "completed"
    start = next(row for row in process.requests if row.get("method") == "thread/start")
    assert start["params"]["config"] == {
        "mcp_servers": {"docs": {"enabled": False}},
        "plugins": {
            "repo-tools@local": {"enabled": False},
            "already-off@local": {"enabled": False},
        },
    }


def test_missing_item_completion_is_not_a_valid_benchmark(monkeypatch: Any) -> None:
    process, _ = _install_fake(monkeypatch, "missing_completion")
    result = benchmark.run(timeout=2)

    assert result["status"] == "incomplete"
    assert result["tps"] is None
    assert result["measurement_method"] is None
    assert "missing_item_completion" in result["diagnostics"]
    assert process.poll() == 0


def test_explicit_effort_overrides_profile_default(monkeypatch: Any) -> None:
    process, _ = _install_fake(monkeypatch, "reasoning")
    result = benchmark.run(effort="low", timeout=2)

    assert result["status"] == "completed"
    assert result["effort"] == "low"
    assert "effort_rerouted" not in result["diagnostics"]
    start = next(row for row in process.requests if row.get("method") == "thread/start")
    assert start["params"]["config"]["model_reasoning_effort"] == "low"


def test_default_effort_preserves_configured_ultra(monkeypatch: Any) -> None:
    process, _ = _install_fake(monkeypatch, "reasoning")
    result = benchmark.run(timeout=2)

    assert result["status"] == "completed"
    assert result["effort"] == "ultra"
    start = next(row for row in process.requests if row.get("method") == "thread/start")
    assert "model_reasoning_effort" not in start["params"]["config"]
    turn = next(row for row in process.requests if row.get("method") == "turn/start")
    assert "effort" not in turn["params"]


def test_missing_item_timestamp_fails_closed(monkeypatch: Any) -> None:
    _install_fake(monkeypatch, "missing_timing")
    result = benchmark.run(timeout=2)

    assert result["status"] == "incomplete"
    assert result["tps"] is None
    assert "missing_item_timing" in result["diagnostics"]


def test_reasoning_tokens_require_reasoning_item_timing(monkeypatch: Any) -> None:
    _install_fake(monkeypatch, "missing_reasoning")
    result = benchmark.run(timeout=2)

    assert result["status"] == "incomplete"
    assert result["tps"] is None
    assert "missing_reasoning_timing" in result["diagnostics"]


def test_each_item_timing_must_be_ordered(monkeypatch: Any) -> None:
    _install_fake(monkeypatch, "reverse_item")
    result = benchmark.run(timeout=2)

    assert result["status"] == "incomplete"
    assert result["tps"] is None
    assert "invalid_item_timing" in result["diagnostics"]
    assert "missing_reasoning_timing" not in result["diagnostics"]


def test_missing_same_turn_usage_fails_closed(monkeypatch: Any) -> None:
    _install_fake(monkeypatch, "missing_usage")
    result = benchmark.run(timeout=2)

    assert result["status"] == "incomplete"
    assert result["output_tokens"] is None
    assert result["tps"] is None
    assert "missing_usage" in result["diagnostics"]


def test_tool_item_is_cancelled_and_excluded(monkeypatch: Any) -> None:
    process, _ = _install_fake(monkeypatch, "tool")
    result = benchmark.run(timeout=2)

    assert result["status"] == "incomplete"
    assert result["tps"] is None
    assert "tool_used" in result["diagnostics"]
    assert any(row.get("method") == "turn/interrupt" for row in process.requests)


def test_timeout_and_rpc_error_return_safe_codes(monkeypatch: Any) -> None:
    timeout_process, _ = _install_fake(monkeypatch, "timeout")
    timeout_result = benchmark.run(timeout=0.05)
    assert timeout_result["status"] == "timeout"
    assert timeout_result["diagnostics"] == ["timeout"]
    assert timeout_process.poll() is not None

    _install_fake(monkeypatch, "rpc_error")
    error_result = benchmark.run(timeout=1)
    assert error_result["status"] == "failed"
    assert error_result["diagnostics"] == ["initialize_failed"]
    assert "private server error" not in repr(error_result)
