"""Tests for session context propagation via ContextVar."""

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.tracing.session import (
    get_current_session_id,
    reset_current_session_id,
    set_current_session_id,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global client and session context between tests."""
    original_client = client_module._global_client
    original_flag = client_module._auto_init_attempted
    client_module._global_client = None
    client_module._auto_init_attempted = False
    token = set_current_session_id(None)
    yield
    reset_current_session_id(token)
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original_client
    client_module._auto_init_attempted = original_flag


class TestSessionContextVar:
    def test_default_is_none(self):
        assert get_current_session_id() is None

    def test_set_and_get(self):
        token = set_current_session_id("sess-1")
        assert get_current_session_id() == "sess-1"
        reset_current_session_id(token)
        assert get_current_session_id() is None

    def test_nested_set_and_reset(self):
        token1 = set_current_session_id("outer")
        assert get_current_session_id() == "outer"
        token2 = set_current_session_id("inner")
        assert get_current_session_id() == "inner"
        reset_current_session_id(token2)
        assert get_current_session_id() == "outer"
        reset_current_session_id(token1)
        assert get_current_session_id() is None


class TestSetSession:
    def test_set_session_changes_context(self):
        pandaprobe.set_session("sess-abc")
        assert get_current_session_id() == "sess-abc"


class TestSessionContextManager:
    def test_scoped_session(self):
        assert get_current_session_id() is None
        with pandaprobe.session("scoped-1"):
            assert get_current_session_id() == "scoped-1"
        assert get_current_session_id() is None

    def test_nested_sessions(self):
        with pandaprobe.session("outer"):
            assert get_current_session_id() == "outer"
            with pandaprobe.session("inner"):
                assert get_current_session_id() == "inner"
            assert get_current_session_id() == "outer"
        assert get_current_session_id() is None


class TestSessionPropagation:
    @respx.mock
    def test_trace_context_picks_up_session(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client = pandaprobe.Client(
            api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0
        )
        pandaprobe.set_session("ctx-session")
        with client.trace("t") as t:
            pass
        assert t._session_id == "ctx-session"
        client.shutdown()

    @respx.mock
    def test_explicit_session_overrides_context(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client = pandaprobe.Client(
            api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0
        )
        pandaprobe.set_session("ctx-session")
        with client.trace("t", session_id="explicit-session") as t:
            pass
        assert t._session_id == "explicit-session"
        client.shutdown()

    @respx.mock
    def test_decorator_picks_up_session(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        pandaprobe.init(
            api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0
        )

        @pandaprobe.trace(name="decorated")
        def my_func():
            return 42

        with pandaprobe.session("decorator-session"):
            my_func()

    @respx.mock
    def test_langgraph_handler_picks_up_session(self):
        from uuid import uuid4

        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        pandaprobe.init(
            api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0
        )
        from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

        pandaprobe.set_session("lg-session")
        handler = LangGraphCallbackHandler()
        root_id = uuid4()
        handler.on_chain_start({"name": "Graph"}, {"input": "hi"}, run_id=root_id)
        handler.on_chain_end({"output": "bye"}, run_id=root_id)
