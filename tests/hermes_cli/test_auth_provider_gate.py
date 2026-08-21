"""Tests for is_provider_explicitly_configured()."""

import json
import pytest


def _write_config(tmp_path, config: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    import yaml
    (hermes_home / "config.yaml").write_text(yaml.dump(config))


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


@pytest.fixture(autouse=True)
def _clean_anthropic_env(monkeypatch):
    """Strip Anthropic env vars so CI secrets don't leak into tests."""
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)






def test_ambient_pool_source_does_not_count_as_explicit(tmp_path, monkeypatch):
    """gh_cli-seeded Copilot pool entries are ambient, not explicit config (#56974)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": None,
        "credential_pool": {
            "copilot": [{
                "id": "abc123",
                "source": "gh_cli",
                "auth_type": "api_key",
                "access_token": "ghu_sometoken",
            }],
        },
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("copilot") is False


def test_external_process_provider_is_explicit_by_construction(tmp_path, monkeypatch):
    """ACP agent backends count as explicit without being the default model.

    An external_process provider only exists because the user installed its
    plugin and the agent's ACP adapter binary. It holds no credential of its
    own, so the auth.json / env-var / credential-pool checks can never pass
    for it. Without an explicit branch it stays permanently invisible to
    explicit-only pickers unless promoted to the user's default provider.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"model": {"provider": "minimax", "default": "MiniMax-M3"}})
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "active_provider": "minimax"})

    from hermes_cli.auth import PROVIDER_REGISTRY, is_provider_explicitly_configured

    external = [
        slug for slug, cfg in PROVIDER_REGISTRY.items()
        if getattr(cfg, "auth_type", "") == "external_process"
    ]
    assert external, "expected at least one external_process provider in the registry"
    for slug in external:
        assert is_provider_explicitly_configured(slug) is True, slug

    # An api_key provider with no credential must still be excluded, so the
    # ambient-credential gate this function exists for is unaffected.
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert is_provider_explicitly_configured("copilot") is False


def test_external_process_gate_falls_back_to_plugin_registry(tmp_path, monkeypatch):
    """The second half of the external_process gate, pinned on its own.

    PROVIDER_REGISTRY is the manually-maintained table. A plugin-registered
    ACP provider may be reachable ONLY through the plugin registry
    (``hermes_cli.providers.get_provider``) in a long-lived process where
    derivation ran after the table was built — the desktop ``hermes serve``
    backends. Short-lived CLI invocations always have derivation behind
    them, so a fallback-less gate passes every CLI test and silently fails in
    exactly the processes that matter most. This test sabotages nothing but
    the table: the provider is absent from PROVIDER_REGISTRY and present in
    the plugin registry, and must still count as explicit.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {"model": {"provider": "minimax", "default": "MiniMax-M3"}})
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "active_provider": "minimax"})

    from types import SimpleNamespace
    import hermes_cli.auth as auth_mod
    import hermes_cli.providers as providers_mod

    slug = "plugin-only-acp"
    assert slug not in auth_mod.PROVIDER_REGISTRY

    real_get_provider = providers_mod.get_provider

    def _fake_get_provider(provider_id):
        if provider_id == slug:
            return SimpleNamespace(auth_type="external_process", base_url="acp://plugin-only")
        return real_get_provider(provider_id)

    monkeypatch.setattr(providers_mod, "get_provider", _fake_get_provider)
    assert auth_mod.is_provider_explicitly_configured(slug) is True

    # Sever the fallback's reach: a plugin-registry miss must NOT be explicit.
    monkeypatch.setattr(providers_mod, "get_provider", lambda pid: None)
    assert auth_mod.is_provider_explicitly_configured(slug) is False


def test_returns_true_when_moa_reference_slot_uses_provider(tmp_path, monkeypatch):
    """MoA advisor slots are explicit provider selections for auth gating."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_config(tmp_path, {
        "model": {"provider": "openai-codex", "default": "gpt-5.5"},
        "moa": {
            "presets": {
                "default": {
                    "reference_models": [
                        {"provider": "anthropic", "model": "claude-opus-4-8"},
                        {"provider": "opencode-go", "model": "glm-5.2"},
                    ],
                    "aggregator": {"provider": "openai-codex", "model": "gpt-5.5"},
                }
            }
        },
    })
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "active_provider": "openai-codex"})

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("anthropic") is True


def test_stale_env_pool_entry_does_not_count_when_var_unset(tmp_path, monkeypatch):
    """An env-seeded pool entry left in auth.json after the env var was removed
    must not mark the provider configured (#55790): the picker showed removed
    providers forever because the record existed even though no secret resolves."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _write_auth_store(tmp_path, {
        "version": 1,
        "providers": {},
        "active_provider": None,
        "credential_pool": {
            "deepseek": [{
                "id": "aaa111",
                "source": "env:DEEPSEEK_API_KEY",
                "auth_type": "api_key",
            }],
        },
    })

    from hermes_cli.auth import is_provider_explicitly_configured
    assert is_provider_explicitly_configured("deepseek") is False






