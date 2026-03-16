"""Tests for pandaprobe.tracing (context, span, session)."""

import pytest
import respx
import httpx

from pandaprobe.client import Client
from pandaprobe.schemas import SpanStatusCode, TraceStatus
from pandaprobe.tracing.context import get_current_trace, get_span_stack


@pytest.fixture
def client():
    c = Client(api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0)
    yield c
    c.shutdown()


class TestTraceContext:
    @respx.mock
    def test_basic_trace(self, client):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        with client.trace("test-trace", input={"q": "hello"}) as t:
            assert get_current_trace() is t
            t.set_output({"a": "world"})

        assert get_current_trace() is None

    @respx.mock
    def test_trace_with_spans(self, client):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        with client.trace("multi-span") as t:
            with t.span("span-1", kind="LLM") as s1:
                s1.set_output("result1")
                s1.set_token_usage(prompt_tokens=5, completion_tokens=10)
            with t.span("span-2", kind="TOOL") as s2:
                s2.set_output("result2")

        assert len(t._spans) == 2

    @respx.mock
    def test_nested_spans(self, client):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        with client.trace("nested") as t:
            with t.span("parent", kind="AGENT") as parent:
                parent_id = parent.span_id
                with t.span("child", kind="LLM") as child:
                    assert child._parent_span_id == parent_id

    @respx.mock
    def test_error_captured(self, client):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        with pytest.raises(ValueError, match="boom"):
            with client.trace("error-trace") as t:
                raise ValueError("boom")

        assert t._status == TraceStatus.ERROR
        assert t._error == "boom"


class TestSpanContext:
    @respx.mock
    def test_span_auto_timing(self, client):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        with client.trace("t") as t:
            with t.span("s"):
                pass
        span_data = t._spans[0]
        assert span_data.started_at is not None
        assert span_data.ended_at is not None
        assert span_data.status == SpanStatusCode.OK

    @respx.mock
    def test_span_error_captured(self, client):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        with pytest.raises(RuntimeError):
            with client.trace("t") as t:
                with t.span("s"):
                    raise RuntimeError("span-error")

        span_data = t._spans[0]
        assert span_data.status == SpanStatusCode.ERROR
        assert span_data.error == "span-error"


class TestSpanStack:
    @respx.mock
    def test_stack_isolation(self, client):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        assert get_span_stack() == []
        with client.trace("t") as t:
            assert get_span_stack() == []
            with t.span("s1") as s1:
                assert get_span_stack() == [s1.span_id]
                with t.span("s2") as s2:
                    assert get_span_stack() == [s1.span_id, s2.span_id]
                assert get_span_stack() == [s1.span_id]
            assert get_span_stack() == []


class TestSessionManager:
    def test_session_passes_session_id(self, client):
        session = client.session("conv-123")
        assert session.session_id == "conv-123"
