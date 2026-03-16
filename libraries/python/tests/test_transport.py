"""Tests for pandaprobe.transport."""

import httpx
import pytest
import respx

from pandaprobe.config import SdkConfig
from pandaprobe.transport import Transport


@pytest.fixture
def config():
    return SdkConfig(
        api_key="sk_pp_test",
        project_name="test-project",
        endpoint="http://testserver",
        batch_size=2,
        flush_interval=0.5,
        max_queue_size=100,
    )


@pytest.fixture
def disabled_config():
    return SdkConfig(enabled=False)


class TestTransportBasics:
    def test_disabled_transport_does_nothing(self, disabled_config):
        transport = Transport(disabled_config)
        transport.enqueue_trace({"name": "test"})
        transport.flush(timeout=1.0)
        transport.shutdown()

    @respx.mock
    def test_enqueue_and_flush_trace(self, config):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={"trace_id": "abc"}))
        transport = Transport(config)
        try:
            transport.enqueue_trace({"name": "test-trace", "started_at": "2025-01-01T00:00:00Z"})
            transport.flush(timeout=5.0)
            assert route.call_count >= 1
        finally:
            transport.shutdown()

    @respx.mock
    def test_enqueue_score(self, config):
        route = respx.post("http://testserver/evaluations/trace-scores").mock(
            return_value=httpx.Response(201, json={"id": "s1"})
        )
        transport = Transport(config)
        try:
            transport.enqueue_score({"trace_id": "abc", "name": "metric", "value": "0.9"})
            transport.flush(timeout=5.0)
            assert route.call_count >= 1
        finally:
            transport.shutdown()

    @respx.mock
    def test_batch_flushing(self, config):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={"trace_id": "x"}))
        transport = Transport(config)
        try:
            for i in range(5):
                transport.enqueue_trace({"name": f"trace-{i}", "started_at": "2025-01-01T00:00:00Z"})
            transport.flush(timeout=5.0)
            assert route.call_count == 5
        finally:
            transport.shutdown()

    @respx.mock
    def test_headers_sent(self, config):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        transport = Transport(config)
        try:
            transport.enqueue_trace({"name": "t"})
            transport.flush(timeout=5.0)
            assert route.call_count >= 1
            req = route.calls[0].request
            assert req.headers["X-API-Key"] == "sk_pp_test"
            assert req.headers["X-Project-Name"] == "test-project"
            assert "X-Request-ID" in req.headers
            assert "pandaprobe-python/" in req.headers["User-Agent"]
        finally:
            transport.shutdown()

    @respx.mock
    def test_update_trace(self, config):
        route = respx.patch("http://testserver/traces/tid-1").mock(return_value=httpx.Response(200, json={}))
        transport = Transport(config)
        try:
            transport.enqueue_update_trace("tid-1", {"status": "COMPLETED"})
            transport.flush(timeout=5.0)
            assert route.call_count >= 1
        finally:
            transport.shutdown()

    @respx.mock
    def test_update_span(self, config):
        route = respx.patch("http://testserver/traces/tid-1/spans/sid-1").mock(
            return_value=httpx.Response(200, json={})
        )
        transport = Transport(config)
        try:
            transport.enqueue_update_span("tid-1", "sid-1", {"status": "OK"})
            transport.flush(timeout=5.0)
            assert route.call_count >= 1
        finally:
            transport.shutdown()

    @respx.mock
    def test_add_spans(self, config):
        route = respx.post("http://testserver/traces/tid-1/spans").mock(
            return_value=httpx.Response(201, json={"span_ids": ["s1"]})
        )
        transport = Transport(config)
        try:
            transport.enqueue_spans("tid-1", [{"name": "s1", "started_at": "2025-01-01T00:00:00Z"}])
            transport.flush(timeout=5.0)
            assert route.call_count >= 1
        finally:
            transport.shutdown()


class TestRetryBehavior:
    @respx.mock
    def test_no_retry_on_401(self, config):
        route = respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )
        transport = Transport(config)
        try:
            transport.enqueue_trace({"name": "t"})
            transport.flush(timeout=5.0)
            assert route.call_count == 1
        finally:
            transport.shutdown()

    @respx.mock
    def test_no_retry_on_422(self, config):
        route = respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(422, json={"detail": "Validation error"})
        )
        transport = Transport(config)
        try:
            transport.enqueue_trace({"name": "t"})
            transport.flush(timeout=5.0)
            assert route.call_count == 1
        finally:
            transport.shutdown()


class TestErrorCallback:
    @respx.mock
    def test_on_error_callback_invoked(self, config):
        errors: list[Exception] = []
        respx.post("http://testserver/traces").mock(side_effect=httpx.ConnectError("refused"))
        transport = Transport(config, on_error=errors.append)
        try:
            transport.enqueue_trace({"name": "t"})
            transport.flush(timeout=10.0)
            assert len(errors) >= 1
        finally:
            transport.shutdown()
