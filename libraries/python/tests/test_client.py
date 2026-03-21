"""Tests for pandaprobe.client."""

import pandaprobe
import pandaprobe.client as client_module
import pytest
from pandaprobe.client import Client, get_client, init


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global client and auto-init flag between tests."""
    original_client = client_module._global_client
    original_flag = client_module._auto_init_attempted
    client_module._global_client = None
    client_module._auto_init_attempted = False
    yield
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original_client
    client_module._auto_init_attempted = original_flag


class TestClient:
    def test_client_disabled(self):
        c = Client(enabled=False)
        assert c.enabled is False
        c.log_trace(None)
        c.flush()
        c.shutdown()

    def test_client_enabled(self):
        c = Client(api_key="sk_pp_test", project_name="proj")
        assert c.enabled is True
        c.shutdown()

    def test_client_missing_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            Client(project_name="proj")

    def test_client_missing_project_raises(self):
        with pytest.raises(ValueError, match="project name"):
            Client(api_key="sk_pp_x")


class TestSingleton:
    def test_init_sets_global(self):
        assert get_client() is None
        c = init(api_key="sk_pp_a", project_name="p")
        assert get_client() is c

    def test_init_replaces_previous(self):
        c1 = init(api_key="sk_pp_a", project_name="p")
        c2 = init(api_key="sk_pp_b", project_name="q")
        assert get_client() is c2
        assert get_client() is not c1


class TestAutoInit:
    def test_auto_inits_from_env(self, monkeypatch):
        monkeypatch.setenv("PANDAPROBE_API_KEY", "sk_pp_auto")
        monkeypatch.setenv("PANDAPROBE_PROJECT_NAME", "auto-proj")
        monkeypatch.setenv("PANDAPROBE_ENABLED", "true")
        c = get_client()
        assert c is not None
        assert c.enabled is True

    def test_returns_none_without_env(self, monkeypatch):
        monkeypatch.delenv("PANDAPROBE_API_KEY", raising=False)
        monkeypatch.delenv("PANDAPROBE_PROJECT_NAME", raising=False)
        c = get_client()
        assert c is None

    def test_disabled_via_enabled_env(self, monkeypatch):
        monkeypatch.setenv("PANDAPROBE_API_KEY", "sk_pp_auto")
        monkeypatch.setenv("PANDAPROBE_PROJECT_NAME", "auto-proj")
        monkeypatch.setenv("PANDAPROBE_ENABLED", "false")
        c = get_client()
        assert c is None

    def test_only_attempted_once(self, monkeypatch):
        monkeypatch.delenv("PANDAPROBE_API_KEY", raising=False)
        monkeypatch.delenv("PANDAPROBE_PROJECT_NAME", raising=False)
        assert get_client() is None
        assert client_module._auto_init_attempted is True
        monkeypatch.setenv("PANDAPROBE_API_KEY", "sk_pp_late")
        monkeypatch.setenv("PANDAPROBE_PROJECT_NAME", "late-proj")
        assert get_client() is None

    def test_explicit_init_overrides_auto(self, monkeypatch):
        monkeypatch.setenv("PANDAPROBE_API_KEY", "sk_pp_auto")
        monkeypatch.setenv("PANDAPROBE_PROJECT_NAME", "auto-proj")
        auto_client = get_client()
        assert auto_client is not None
        explicit_client = init(api_key="sk_pp_explicit", project_name="explicit-proj")
        assert get_client() is explicit_client
        assert get_client() is not auto_client


class TestScore:
    def test_score_disabled_noop(self):
        c = Client(enabled=False)
        c.score("00000000-0000-0000-0000-000000000000", "metric", "0.5")


class TestOnError:
    def test_error_callback_registered(self):
        c = Client(api_key="sk_pp_x", project_name="p")
        errors = []
        c.on_error(errors.append)
        c._dispatch_error(RuntimeError("test"))
        assert len(errors) == 1
        c.shutdown()


class TestModuleLevelImports:
    def test_public_api(self):
        assert hasattr(pandaprobe, "Client")
        assert hasattr(pandaprobe, "init")
        assert hasattr(pandaprobe, "get_client")
        assert hasattr(pandaprobe, "trace")
        assert hasattr(pandaprobe, "span")
        assert hasattr(pandaprobe, "__version__")
        assert hasattr(pandaprobe, "flush")
        assert hasattr(pandaprobe, "shutdown")
        assert hasattr(pandaprobe, "score")
        assert hasattr(pandaprobe, "set_session")
        assert hasattr(pandaprobe, "session")
        assert hasattr(pandaprobe, "set_user")
        assert hasattr(pandaprobe, "user")
