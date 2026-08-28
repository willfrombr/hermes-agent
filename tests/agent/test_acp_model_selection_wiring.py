"""End-to-end wiring for picker-driven model selection over a fake ACP server.

Review ask on the model-selection PR: nothing exercised the actual wire —
``session/set_config_option`` emission, the confirmed-value read-back, and
``resolved_model`` reaching ``completion.model``. This drives ``_run_prompt``
and ``_create_chat_completion`` against a real subprocess speaking
line-delimited JSON-RPC, records every request it receives, and asserts the
emitted protocol rather than internal state.
"""

import json
import sys
import textwrap

import pytest

from agent.acp_client import ACPClient

FAKE_SERVER = textwrap.dedent(
    """
    import json, sys

    log_path = sys.argv[1]
    requests = []

    def log(entry):
        requests.append(entry)
        with open(log_path, "w") as f:
            json.dump(requests, f)

    current = "sonnet"
    def options():
        return [{
            "id": "model", "category": "model", "currentValue": current,
            "options": [
                {"value": "sonnet", "name": "Claude Sonnet 5"},
                {"value": "opus[1m]", "name": "Claude Opus 5"},
            ],
        }]

    for line in sys.stdin:
        msg = json.loads(line)
        log({"method": msg.get("method"), "params": msg.get("params")})
        rid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            result = {"protocolVersion": 1}
        elif method == "session/new":
            result = {"sessionId": "s1", "configOptions": options()}
        elif method == "session/set_config_option":
            current = msg["params"]["value"]
            result = {"configOptions": options()}
        elif method == "session/prompt":
            note = {"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "s1",
                "update": {"sessionUpdate": "agent_message_chunk",
                           "content": {"type": "text", "text": "hello from fake"}},
            }}
            sys.stdout.write(json.dumps(note) + "\\n"); sys.stdout.flush()
            result = {"stopReason": "end_turn"}
        else:
            result = {}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\\n")
        sys.stdout.flush()
    """
)


@pytest.fixture()
def fake_agent(tmp_path):
    server = tmp_path / "fake_acp_server.py"
    server.write_text(FAKE_SERVER)
    log = tmp_path / "requests.json"
    client = ACPClient(
        agent_name="claude",
        acp_cwd=str(tmp_path),
        acp_command=sys.executable,
        acp_args=[str(server), str(log)],
    )
    def requests():
        return json.loads(log.read_text())
    return client, requests


def test_explicit_pick_drives_set_config_option_and_labels_result(fake_agent):
    client, requests = fake_agent
    text, reasoning, resolved = client._run_prompt(
        "hi", timeout_seconds=20, model="opus-5"
    )
    assert text == "hello from fake"
    sets = [r for r in requests() if r["method"] == "session/set_config_option"]
    assert len(sets) == 1
    assert sets[0]["params"]["configId"] == "model"
    assert sets[0]["params"]["value"] == "opus[1m]"
    # Confirmed value read back out of the set response, not assumed.
    assert resolved == "opus[1m]"


def test_sentinel_pick_makes_no_set_call_and_reports_current(fake_agent):
    client, requests = fake_agent
    text, _, resolved = client._run_prompt(
        "hi", timeout_seconds=20, model="claude-acp"
    )
    assert text == "hello from fake"
    assert [r for r in requests() if r["method"] == "session/set_config_option"] == []
    assert resolved == "sonnet"


def test_resolved_model_reaches_completion_model(fake_agent):
    client, _ = fake_agent
    completion = client._create_chat_completion(
        model="opus-5",
        messages=[{"role": "user", "content": "hi"}],
        timeout=20,
    )
    assert completion.model == "opus[1m]"
