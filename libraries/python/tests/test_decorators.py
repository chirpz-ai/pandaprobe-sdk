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
        def run_agent(messages: list):
            return {"messages": [{"role": "assistant", "content": "answer"}]}

        result = run_agent([{"role": "user", "content": "hello"}])
        assert result == {"messages": [{"role": "assistant", "content": "answer"}]}

    @respx.mock
    def test_sync_trace_no_parens(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace
        def run_agent(messages: list):
            return {"messages": [{"role": "assistant", "content": "answer"}]}

        result = run_agent([{"role": "user", "content": "hello"}])
        assert result == {"messages": [{"role": "assistant", "content": "answer"}]}

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_trace(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace(name="async-agent")
        async def run_agent(messages: list):
            return {"messages": [{"role": "assistant", "content": "async answer"}]}

        result = await run_agent([{"role": "user", "content": "hello"}])
        assert result == {"messages": [{"role": "assistant", "content": "async answer"}]}

    @respx.mock
    def test_trace_captures_exception(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace
        def failing(messages: list):
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            failing([{"role": "user", "content": "x"}])

    @respx.mock
    def test_trace_invalid_input_raises(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace(name="bad-input")
        def bad_fn(query: str):
            return {"messages": [{"role": "assistant", "content": "ok"}]}

        with pytest.raises(ValueError, match="trace input"):
            bad_fn("raw string")

    @respx.mock
    def test_trace_invalid_output_raises(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace(name="bad-output")
        def bad_fn(messages: list):
            return "raw string"

        with pytest.raises(ValueError, match="trace output"):
            bad_fn([{"role": "user", "content": "hello"}])

    @respx.mock
    def test_trace_trims_input_to_last_user_message(self):
        """Full conversation history should be trimmed to the last user message."""
        captured = {}
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace(name="multi-turn")
        def chat(messages: list):
            return {"messages": [{"role": "assistant", "content": "reply"}]}

        original_trace = pandaprobe.client.get_client().trace

        def spy_trace(name, **kw):
            captured["input"] = kw.get("input")
            return original_trace(name, **kw)

        pandaprobe.client.get_client().trace = spy_trace

        conversation = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
        chat(conversation)

        assert captured["input"] == {
            "messages": [{"role": "user", "content": "second question"}]
        }


class TestSpanDecorator:
    @respx.mock
    def test_span_inside_trace(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @span(name="llm-call", kind="LLM")
        def call_llm(messages: list):
            return {"messages": [{"role": "assistant", "content": "response"}]}

        @trace(name="agent")
        def run(messages: list):
            return call_llm(messages)

        result = run([{"role": "user", "content": "test"}])
        assert result == {"messages": [{"role": "assistant", "content": "response"}]}

    @respx.mock
    def test_span_without_trace_runs_normally(self):
        """If no trace is active, the span decorator should just call the function."""

        @span(name="standalone")
        def do_something():
            return 42

        assert do_something() == 42

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_span(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @span(name="async-llm", kind="LLM")
        async def call_llm(messages: list):
            return {"messages": [{"role": "assistant", "content": "async-response"}]}

        @trace(name="async-agent")
        async def run(messages: list):
            return await call_llm(messages)

        result = await run([{"role": "user", "content": "test"}])
        assert result == {"messages": [{"role": "assistant", "content": "async-response"}]}

    @respx.mock
    def test_default_name(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @trace
        def my_function(messages: list):
            return {"messages": [{"role": "assistant", "content": "ok"}]}

        assert my_function.__name__ == "my_function"
        assert my_function([{"role": "user", "content": "hi"}]) == {
            "messages": [{"role": "assistant", "content": "ok"}]
        }

    @respx.mock
    def test_non_llm_span_accepts_arbitrary_io(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        @span(name="tool-call", kind="TOOL")
        def run_tool(query: str):
            return "arbitrary output"

        @trace(name="agent")
        def run(messages: list):
            run_tool("query")
            return {"messages": [{"role": "assistant", "content": "ok"}]}

        result = run([{"role": "user", "content": "hi"}])
        assert result == {"messages": [{"role": "assistant", "content": "ok"}]}
