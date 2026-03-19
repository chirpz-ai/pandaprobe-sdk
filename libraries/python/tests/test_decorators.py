"""Tests for pandaprobe.decorators."""

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.decorators import span, trace


@pytest.fixture(autouse=True)
def _setup_client():
    """Set up and tear down a global client for decorator tests."""
    original_client = client_module._global_client
    original_flag = client_module._auto_init_attempted
    pandaprobe.init(api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0)
    yield
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original_client
    client_module._auto_init_attempted = original_flag


class TestTraceDecorator:
    @respx.mock
    def test_sync_trace(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace(name="my-agent")
        def run_agent(query: str):
            return f"answer to {query}"

        result = run_agent("hello")
        assert result == "answer to hello"

    @respx.mock
    def test_sync_trace_no_parens(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace
        def run_agent(query: str):
            return f"answer to {query}"

        result = run_agent("hello")
        assert result == "answer to hello"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_trace(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace(name="async-agent")
        async def run_agent(query: str):
            return f"async answer to {query}"

        result = await run_agent("hello")
        assert result == "async answer to hello"

    @respx.mock
    def test_trace_captures_exception(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace
        def failing():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            failing()


class TestSpanDecorator:
    @respx.mock
    def test_span_inside_trace(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @span(name="llm-call", kind="LLM")
        def call_llm(prompt: str):
            return "response"

        @trace(name="agent")
        def run(query: str):
            return call_llm(query)

        result = run("test")
        assert result == "response"

    @respx.mock
    def test_span_without_trace_runs_normally(self):
        """If no trace is active, the span decorator should just call the function."""

        @span(name="standalone")
        def do_something():
            return 42

        # No active trace → function should run but no span created
        assert do_something() == 42

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_span(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @span(name="async-llm", kind="LLM")
        async def call_llm(prompt: str):
            return "async-response"

        @trace(name="async-agent")
        async def run(query: str):
            return await call_llm(query)

        result = await run("test")
        assert result == "async-response"

    @respx.mock
    def test_default_name(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace
        def my_function():
            return True

        assert my_function.__name__ == "my_function"
        assert my_function() is True
