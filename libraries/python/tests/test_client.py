"""Tests for pandaprobe.client."""

import pandaprobe
from pandaprobe.client import Client, get_client, init
import pandaprobe.client as client_module
import pytest


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global client between tests."""
    original = client_module._global_client
    client_module._global_client = None
    yield
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original


class TestClient:
    def test_client_disabled(self):
        c = Client(enabled=False)
        assert c.enabled is False
        c.log_trace(None)  # should not raise
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


class TestSession:
    def test_session_returns_manager(self):
        c = Client(api_key="sk_pp_x", project_name="p")
        session = c.session("sess-1")
        assert session.session_id == "sess-1"
        c.shutdown()


class TestModuleLevelImports:
    def test_public_api(self):
        assert hasattr(pandaprobe, "Client")
        assert hasattr(pandaprobe, "init")
        assert hasattr(pandaprobe, "get_client")
        assert hasattr(pandaprobe, "trace")
        assert hasattr(pandaprobe, "span")
        assert hasattr(pandaprobe, "__version__")
