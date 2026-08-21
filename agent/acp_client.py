"""OpenAI-compatible shim that forwards Hermes requests to an ACP agent.

Generalization of the original Copilot-only client (issue #5257): any agent
registered in :mod:`agent.acp_agent_registry` — Claude Code, Codex CLI,
Gemini CLI, Qwen Code, GitHub Copilot, or anything reachable via a
``HERMES_ACP_{NAME}_COMMAND`` override — can serve as a chat-style backend
through its official ACP adapter. Each request starts a short-lived ACP
session, sends the formatted conversation as a single prompt, collects text
chunks, and converts the result back into the minimal shape Hermes expects
from an OpenAI client.

The protocol loop, filesystem callbacks, and tool-call extraction are
unchanged from the battle-tested Copilot client; only the launch command,
marker URL, and messages are parameterized.

Permission requests are NOT inherited from that client. Copilot CLI rarely
emits ``session/request_permission``, so its blanket-deny posture went
unnoticed there — but Claude Code gates ordinary state-changing commands
that way, and a blanket deny surfaces to the agent as "Tool use aborted"
for every one of them. Requests are instead routed through Hermes's own
approval policy; see :meth:`ACPClient._decide_permission` and the
``approvals.acp_mode`` config key.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.acp_agent_registry import (
    agent_display_name,
    agent_env_unset,
    agent_install_hint,
    normalize_agent_name,
    resolve_agent_launch,
)
from agent.file_safety import (
    get_read_block_error,
    get_write_denied_error,
    is_write_approval_required,
)
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

ACP_MARKER_PREFIX = "acp://"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)


def extract_agent_from_url(base_url: Any) -> str | None:
    """Return the agent name from an ``acp://{agent}`` URL, else ``None``."""
    text = str(base_url or "").strip()
    if not text.lower().startswith(ACP_MARKER_PREFIX):
        return None
    name = normalize_agent_name(text[len(ACP_MARKER_PREFIX):].split("/", 1)[0])
    return name or None


def marker_base_url(agent_name: str) -> str:
    return f"{ACP_MARKER_PREFIX}{normalize_agent_name(agent_name)}"


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env(env_unset: tuple[str, ...] = ()) -> dict[str, str]:
    # ACP agents are model-driving CLI executors: they legitimately need LLM
    # provider credentials. Route through the central helper so Tier-1 secrets
    # (gateway bot tokens, GitHub auth, infra) are still stripped (#29157).
    env = hermes_subprocess_env(inherit_credentials=True)
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    # Per-agent session markers to strip (e.g. the Claude Code bridge won't
    # launch inside a parent Claude Code session) — declared on the registry
    # entry so this stays agent-agnostic.
    for marker in env_unset:
        env.pop(marker, None)
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "cancelled",
            }
        },
    }


def _permission_selected(message_id: Any, option_id: str) -> dict[str, Any]:
    """Answer a permission request by selecting one of its offered options."""
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "selected",
                "optionId": option_id,
            }
        },
    }


# ACP option kinds, in the order we prefer them for each verdict. Agents only
# have to offer a subset, so we pick the first kind actually on offer and fall
# back to a "cancelled" outcome when none matches.
_ALLOW_KINDS = ("allow_once", "allow_always")
_REJECT_KINDS = ("reject_once", "reject_always")


def _select_option(options: Any, *, allow: bool) -> str | None:
    """Return the option id matching *allow*, or None when none is offered."""
    if not isinstance(options, list):
        return None
    wanted = _ALLOW_KINDS if allow else _REJECT_KINDS
    for kind in wanted:
        for option in options:
            if not isinstance(option, dict):
                continue
            if str(option.get("kind") or "") == kind:
                option_id = str(option.get("optionId") or option.get("option_id") or "")
                if option_id:
                    return option_id
    return None


def _extract_permission_command(tool_call: Any) -> tuple[str, str]:
    """Best-effort ``(command, description)`` from a permission tool call.

    ACP does not mandate a payload shape for the thing being approved. The
    richest form is ``rawInput`` (what Hermes itself emits when acting as an
    ACP server — see ``acp_adapter/permissions.py``); otherwise fall back to
    the human-readable title so the approval prompt still says something
    meaningful.
    """
    if not isinstance(tool_call, dict):
        return "", ""
    raw = tool_call.get("rawInput") or tool_call.get("raw_input") or {}
    command = ""
    description = ""
    if isinstance(raw, dict):
        command = str(raw.get("command") or "").strip()
        description = str(raw.get("description") or "").strip()
    title = str(tool_call.get("title") or "").strip()
    if not command:
        command = title
    if not description:
        description = title if title != command else str(tool_call.get("kind") or "").strip()
    return command, description


_ACP_PERMISSION_MODES = ("bridge", "deny", "allow")


def _acp_permission_mode() -> str:
    """Resolve ``approvals.acp_mode`` (env override wins). Defaults to bridge."""
    override = os.getenv("HERMES_ACP_PERMISSION_MODE", "").strip().lower()
    if override in _ACP_PERMISSION_MODES:
        return override
    try:
        from tools.approval import _get_approval_config

        mode = str(_get_approval_config().get("acp_mode") or "").strip().lower()
    except Exception:  # pragma: no cover - config layer unavailable
        mode = ""
    return mode if mode in _ACP_PERMISSION_MODES else "bridge"


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    # The "Hermes requested model hint" line is NOT appended here any more.
    # _run_prompt owns it, gated on the config-option path not running:
    # once session/set_config_option succeeds, a hint naming the requested
    # id would be a contradictory instruction alongside the model the
    # session is actually serving.
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Use ACP capabilities to complete tasks.",
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
        "If no tool is needed, answer normally.",
    ]

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot ACP response into OpenAI-style stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


def _extract_tool_calls_from_text(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            _build_openai_tool_call(
                call_id=call_id,
                name=fn_name.strip(),
                arguments=fn_args,
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    # Only try bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned



def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "ACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "ACPClient"):
        self.completions = _ACPChatCompletions(client)



# Probe verdicts cached per binary path so repeated prompts against a
# CLI that supports --acp pay the ~50ms --help cost exactly once per
# process. Only definitive verdicts (True/False) are cached; an
# inconclusive probe (binary missing, --help crashed or timed out) is
# not cached so a CLI installed mid-session is picked up.
_ACP_PROBE_CACHE: dict[str, bool] = {}


def _acp_supported(command: str, args: list[str]) -> bool | None:
    """Tri-state probe: does ``command`` accept the ACP args we'd pass?

    Only agents launched with a literal ``--acp`` flag are probed. Every
    other adapter in the registry speaks ACP through a dedicated binary
    (``claude-agent-acp``, ``codex-acp``) or its own subcommand/flag
    (``cursor-agent acp``, ``gemini --experimental-acp``); those have no
    ``--acp`` to advertise and are skipped, so this never gates a plugin
    agent on a flag it was never going to pass.

    For the ones that do: spawning a CLI that doesn't recognize ``--acp``
    exits with code 1 and ``error: unknown option '--acp'`` on stderr,
    after which the parent ACP loop waits the full
    ``child_timeout_seconds`` (default 600s) for stdout that never
    arrives. Ported from the Copilot-specific client (upstream #87308,
    hardened in #87309) when the protocol moved into this module.

    Returns:
      - ``True``  — help text advertises ``--acp``, or the agent doesn't
        use the flag at all; safe to spawn.
      - ``False`` — help ran cleanly but ``--acp`` is absent; spawning
        would hang, so the caller should fast-fail with a clear error.
      - ``None``  — inconclusive (binary missing, --help failed or timed
        out). The caller must fall through to the normal spawn path,
        which surfaces the established "Could not start ... ACP command"
        error with its install hint.
    """
    if "--acp" not in args:
        return True
    cached = _ACP_PROBE_CACHE.get(command)
    if cached is not None:
        return cached
    try:
        probe = subprocess.run(
            [command, "--help"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if probe.returncode != 0:
        # --help itself failed; can't tell anything about --acp.
        return None
    # Match ``--acp`` as a flag in the help text; tolerate spacing and
    # variants like ``[--acp]``.
    verdict = bool(re.search(r"(?:^|[\s\[])--acp(?:[\s=\],]|$)", probe.stdout, re.MULTILINE))
    _ACP_PROBE_CACHE[command] = verdict
    return verdict

class ACPClient:
    """Minimal OpenAI-client-compatible facade for an ACP agent."""

    def __init__(
        self,
        *,
        agent_name: str = "copilot",
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.agent_name = (
            extract_agent_from_url(base_url)
            or normalize_agent_name(agent_name)
            or "copilot"
        )
        self.agent_display_name = agent_display_name(self.agent_name)
        self.api_key = api_key or f"{self.agent_name}-acp"
        self.base_url = base_url or marker_base_url(self.agent_name)
        self._default_headers = dict(default_headers or {})
        explicit_command = acp_command or command
        if explicit_command:
            # Default args still come from the registry so an explicit command
            # path without explicit args keeps the agent's stdio flags
            # (matches the historical Copilot contract).
            self._acp_command = explicit_command
            try:
                _, resolved_args = resolve_agent_launch(self.agent_name)
            except ValueError:
                resolved_args = []
        else:
            self._acp_command, resolved_args = resolve_agent_launch(self.agent_name)
        self._acp_args = list(acp_args or args or resolved_args)
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()

    def _early_exit_error(self, stderr_text: str) -> str | None:
        """Hook: return a custom error message for an early process exit.

        Subclasses (e.g. Copilot's deprecated-CLI detection) can inspect
        *stderr_text* and return a friendlier message; ``None`` falls back
        to the generic one.
        """
        return None

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            tools=tools,
            tool_choice=tool_choice,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            # httpx.Timeout or similar — pick the largest component so the
            # subprocess has enough wall-clock time for the full response.
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text, resolved_model = self._run_prompt(
            prompt_text,
            timeout_seconds=_effective_timeout,
            model=model,
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            # Prefer the agent-confirmed model id so the UI reports what
            # actually served the turn, not merely what was requested.
            model=resolved_model or model or f"{self.agent_name}-acp",
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    @staticmethod
    def _maybe_append_model_hint(
        prompt_text: str,
        model: str | None,
        mapping: "tuple[str, str, str] | None",
        agent_name: str,
    ) -> str:
        """Append the legacy model hint ONLY when the config-option path
        did not run.

        The hint moved here from ``_format_messages_as_prompt``: once
        ``session/set_config_option`` honours an explicit pick, a prompt-text
        hint naming the requested id would sit alongside — and disagree
        with — the model actually serving the session. It still appears for
        agents advertising no model option (``mapping is None`` — e.g.
        Copilot today) and for sentinel picks that keep the agent default,
        which is exactly the pre-existing behaviour on those paths.
        """
        if model and (
            mapping is None
            or ACPClient._is_model_sentinel(model, agent_name)
        ):
            return f"{prompt_text}\nHermes requested model hint: {model}"
        return prompt_text

    @staticmethod
    def _is_model_sentinel(requested: str, agent_name: str) -> bool:
        """A pick that means "keep the agent's default" — no set call runs.

        Shared between option resolution and the prompt-hint gate so the two
        never disagree about what counts as a sentinel.
        """
        req = (requested or "").strip().lower()
        return req in ("", "default", f"{agent_name}-acp", agent_name)

    @staticmethod
    def _resolve_model_option(
        config_options: list[Any],
        requested: str,
        agent_name: str,
    ) -> tuple[str, str, str] | None:
        """Map a Hermes-side model id onto the agent's advertised model option.

        ACP agents MAY advertise a ``SessionConfigOption`` with
        ``category: "model"`` in the ``session/new`` response (stable since
        spec v1). When present, ``session/set_config_option`` switches the
        live session's model — the prompt-text "model hint" the flattened
        prompt carries is ignored by real agents, so this is the only path
        that actually honours the user's picker choice.

        Returns ``(config_id, target_value, current_value)`` or ``None`` when
        no model option is advertised. Sentinel ids ("default", "<agent>-acp",
        empty) resolve to the agent's current default without a set call.
        Raises ``RuntimeError`` when the agent advertises models but the
        requested id matches none — a wrong pick must surface, not silently
        fall back to whatever the agent defaults to.
        """
        option = None
        for candidate in config_options or []:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("category") == "model" or candidate.get("id") == "model":
                option = candidate
                break
        if option is None:
            return None

        config_id = str(option.get("id") or "model")
        current = str(option.get("currentValue") or "")
        choices = [c for c in option.get("options") or [] if isinstance(c, dict)]

        req = (requested or "").strip().lower()
        if ACPClient._is_model_sentinel(requested, agent_name):
            return (config_id, current, current)

        for choice in choices:  # exact value, then exact display name
            value = str(choice.get("value") or "")
            name = str(choice.get("name") or "")
            if req == value.lower() or req == name.lower():
                return (config_id, value, current)
        for choice in choices:  # prefix/substring — e.g. "fable-5" -> "claude-fable-5[1m]"
            value = str(choice.get("value") or "").lower()
            name = str(choice.get("name") or "").lower()
            # Both directions: versioned Hermes ids ("opus-5", "haiku-4.5")
            # must reach the adapter's short values ("opus", "haiku"), and
            # short requests ("fable") must reach long values
            # ("claude-fable-5[1m]"). Names count too ("fable-5" vs "Fable").
            #
            # Values may carry a context-window suffix ("opus[1m]"), so the
            # stem test has to be a plain prefix match: "opus-5" -> stem
            # "opus" -> "opus[1m]". Requiring equality or a "-" separator
            # matched only suffix-less values, so "haiku-4.5" and "sonnet-5"
            # resolved while "opus-5" raised.
            #
            # Strip a vendor prefix before stemming, or every "claude-*" id
            # stems to "claude" and silently binds to whichever "claude-"
            # value comes first in the list — "claude-opus-5" resolved to
            # claude-fable-5[1m], a different and pricier model, with no error.
            req_stem = req[len("claude-"):] if req.startswith("claude-") else req
            req_stem = req_stem.split("-", 1)[0]
            if (
                value.startswith(req) or req in value
                or req.startswith(value) or (name and req.startswith(name))
                or (req_stem and (value.startswith(req_stem) or req_stem == name))
            ):
                return (config_id, str(choice.get("value") or ""), current)

        available = ", ".join(str(c.get("value")) for c in choices) or "(none)"
        raise RuntimeError(
            f"Model '{requested}' is not offered by this ACP agent. "
            f"Available: {available}"
        )

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        timeout_seconds: float,
        model: str | None = None,
    ) -> tuple[str, str, str | None]:
        display = self.agent_display_name
        # Fast-fail a CLI that doesn't speak the transport we're about to
        # hand it. Without this, such a CLI exits immediately and the loop
        # below waits ``child_timeout_seconds`` (default 600s) for stdout
        # that never arrives. ``None`` (inconclusive) falls through to the
        # spawn, which raises the established "Could not start" error.
        if _acp_supported(self._acp_command, self._acp_args) is False:
            preview = " ".join(self._acp_args[:3]) if self._acp_args else "(none)"
            raise RuntimeError(
                f"ACP transport not supported by '{self._acp_command}': "
                f"`{preview}` is rejected as an unknown option. This usually "
                f"means the CLI is an older release, or a different tool than "
                f"expected. " + agent_install_hint(self.agent_name)
                + f" You can also override the pair with "
                f"HERMES_ACP_{self.agent_name.upper()}_COMMAND / "
                f"HERMES_ACP_{self.agent_name.upper()}_ARGS."
            )
        try:
            # Hide the console the CLI child would otherwise flash on Windows
            # (#56747). Hide-only — stdio pipes stay intact for the ACP wire.
            from hermes_cli._subprocess_compat import windows_hide_flags

            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace',
                bufsize=1,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(env_unset=agent_env_unset(self.agent_name)),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start {display} ACP command '{self._acp_command}'. "
                + agent_install_hint(self.agent_name)
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError(f"{display} ACP process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(
                        f"{display} ACP {method} failed: {err.get('message') or err}"
                    )
                return msg.get("result")

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                custom = self._early_exit_error(stderr_text)
                if custom:
                    raise RuntimeError(custom)
                raise RuntimeError(f"{display} ACP process exited early: {stderr_text}")
            raise TimeoutError(f"Timed out waiting for {display} ACP response to {method}.")

        try:
            _request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {
                            "readTextFile": True,
                            "writeTextFile": True,
                        }
                    },
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.0.0",
                    },
                },
            )
            session = _request(
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": [],
                },
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError(f"{display} ACP did not return a sessionId.")

            # Honour the picker's model choice through the stable ACP config
            # mechanism when the agent advertises one. Agents without a model
            # option (e.g. Copilot today) keep the legacy prompt-hint path.
            resolved_model: str | None = None
            mapping = self._resolve_model_option(
                session.get("configOptions") or [], model or "", self.agent_name
            )
            if mapping is not None:
                config_id, target_value, current_value = mapping
                resolved_model = current_value or None
                if target_value and target_value != current_value:
                    set_result = _request(
                        "session/set_config_option",
                        {
                            "sessionId": session_id,
                            "configId": config_id,
                            "value": target_value,
                        },
                    ) or {}
                    # The response carries the updated option list; read the
                    # confirmed value back rather than assuming the set stuck.
                    confirmed = self._resolve_model_option(
                        set_result.get("configOptions") or [], "", self.agent_name
                    )
                    resolved_model = (confirmed[2] if confirmed else None) or target_value

            prompt_text = self._maybe_append_model_hint(
                prompt_text, model, mapping, self.agent_name
            )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            )
            return "".join(text_parts), "".join(reasoning_parts), resolved_model
        finally:
            self.close()

    def _decide_permission(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Answer an agent's ``session/request_permission``.

        Modes (``approvals.acp_mode``, or ``HERMES_ACP_PERMISSION_MODE``):

        ``bridge`` (default)
            Route the request through :func:`tools.approval.check_dangerous_command`
            — the same gate ``terminal_tool`` uses before running anything. Safe
            commands pass straight through; dangerous ones honour the user's
            deny rules, allowlists, ``/yolo``, and (in a gateway session) an
            interactive approval prompt.
        ``deny``
            Refuse everything. The historical posture, kept for operators who
            want ACP backends to have no side effects at all.
        ``allow``
            Approve everything. For sandboxed deployments that already trust
            whatever the agent can reach.

        Any failure denies: an ACP backend must never gain more privilege than
        Hermes's own terminal tool because the approval layer misbehaved.
        """
        options = params.get("options")
        try:
            mode = _acp_permission_mode()

            if mode == "deny":
                return _permission_denied(message_id)

            if mode == "allow":
                option_id = _select_option(options, allow=True)
                return (
                    _permission_selected(message_id, option_id)
                    if option_id
                    else _permission_denied(message_id)
                )

            command, description = _extract_permission_command(params.get("toolCall"))
            if not command:
                logger.warning(
                    "%s ACP permission request carried no identifiable command; denying",
                    self.agent_display_name,
                )
                return _permission_denied(message_id)

            from tools.approval import check_dangerous_command

            # env_type="local": the ACP agent is a subprocess of this container,
            # so its commands act on us. Claiming a sandboxed backend here would
            # make _should_skip_container_guards() auto-approve everything.
            verdict = check_dangerous_command(command, "local") or {}
            approved = bool(verdict.get("approved"))
            logger.info(
                "%s ACP permission %s: %s",
                self.agent_display_name,
                "approved" if approved else "denied",
                (verdict.get("message") or description or command)[:200],
            )

            option_id = _select_option(options, allow=approved)
            if option_id:
                return _permission_selected(message_id, option_id)
            # Nothing suitable on offer: "cancelled" is the only safe answer,
            # and is what an unapproved request should get anyway.
            return _permission_denied(message_id)
        except Exception:
            logger.exception(
                "%s ACP permission decision failed; denying",
                self.agent_display_name,
            )
            return _permission_denied(message_id)

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = self._decide_permission(message_id, params)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                denied = get_write_denied_error(str(path))
                if denied:
                    raise PermissionError(denied)
                # Approval-gated paths (e.g. ~/.ssh/config) are not hard-denied
                # for interactive tools, but the ACP bridge has no human channel
                # to confirm the write — fail closed here.
                if is_write_approval_required(str(path)):
                    raise PermissionError(
                        f"Write denied: '{path}' requires interactive approval "
                        "and cannot be written through the ACP file bridge."
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""), encoding="utf-8")
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True


def create_acp_client(*, agent_name: str | None = None, **kwargs: Any) -> ACPClient:
    """Factory: build the right ACP client for *agent_name* / ``base_url``.

    Copilot keeps its dedicated subclass (deprecated-CLI detection); every
    other agent uses the generic :class:`ACPClient`.
    """
    name = (
        normalize_agent_name(agent_name or "")
        or extract_agent_from_url(kwargs.get("base_url"))
        or "copilot"
    )
    if name == "copilot":
        from agent.copilot_acp_client import CopilotACPClient

        return CopilotACPClient(**kwargs)
    return ACPClient(agent_name=name, **kwargs)
