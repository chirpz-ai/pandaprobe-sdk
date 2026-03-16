"""Tests for pandaprobe.schemas."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from pandaprobe.schemas import (
    ScoreData,
    ScoreDataType,
    SpanData,
    SpanKind,
    SpanStatusCode,
    TraceData,
    TraceStatus,
)


class TestSpanData:
    def test_defaults(self):
        span = SpanData(name="test-span", started_at=datetime.now(timezone.utc))
        assert span.kind == SpanKind.OTHER
        assert span.status == SpanStatusCode.UNSET
        assert span.metadata == {}
        assert isinstance(span.span_id, UUID)

    def test_to_api_dict_minimal(self):
        now = datetime(2025, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        span = SpanData(name="s", started_at=now)
        d = span.to_api_dict()
        assert d["name"] == "s"
        assert d["kind"] == "OTHER"
        assert d["status"] == "UNSET"
        assert d["started_at"] == "2025-03-14T12:00:00+00:00"
        assert "parent_span_id" not in d
        assert "model" not in d

    def test_to_api_dict_full(self):
        sid = uuid4()
        psid = uuid4()
        now = datetime(2025, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        span = SpanData(
            span_id=sid,
            parent_span_id=psid,
            name="llm-call",
            kind=SpanKind.LLM,
            status=SpanStatusCode.OK,
            input={"messages": [{"role": "user", "content": "hi"}]},
            output={"content": "hello"},
            model="gpt-4",
            token_usage={"prompt_tokens": 10, "completion_tokens": 20},
            metadata={"key": "val"},
            started_at=now,
            ended_at=now,
            model_parameters={"temperature": 0.7},
            cost={"total": 0.001},
        )
        d = span.to_api_dict()
        assert d["span_id"] == str(sid)
        assert d["parent_span_id"] == str(psid)
        assert d["model"] == "gpt-4"
        assert d["token_usage"]["prompt_tokens"] == 10
        assert d["cost"]["total"] == 0.001


class TestTraceData:
    def test_defaults(self):
        now = datetime.now(timezone.utc)
        trace = TraceData(name="test-trace", started_at=now)
        assert trace.status == TraceStatus.COMPLETED
        assert trace.spans == []
        assert trace.tags == []

    def test_to_api_dict(self):
        now = datetime(2025, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        trace = TraceData(
            name="agent-run",
            status=TraceStatus.COMPLETED,
            input={"query": "hello"},
            output={"answer": "world"},
            started_at=now,
            ended_at=now,
            session_id="sess-1",
            user_id="user-1",
            tags=["production"],
            environment="prod",
            release="1.0",
        )
        d = trace.to_api_dict()
        assert d["name"] == "agent-run"
        assert d["status"] == "COMPLETED"
        assert d["session_id"] == "sess-1"
        assert d["tags"] == ["production"]
        assert d["environment"] == "prod"

    def test_validation_name_required(self):
        with pytest.raises(Exception):
            TraceData(name="", started_at=datetime.now(timezone.utc))


class TestScoreData:
    def test_to_api_dict(self):
        tid = uuid4()
        score = ScoreData(
            trace_id=tid,
            name="user_satisfaction",
            value="0.9",
            data_type=ScoreDataType.NUMERIC,
            reason="thumbs up",
        )
        d = score.to_api_dict()
        assert d["trace_id"] == str(tid)
        assert d["name"] == "user_satisfaction"
        assert d["value"] == "0.9"
        assert d["data_type"] == "NUMERIC"
        assert d["source"] == "PROGRAMMATIC"
        assert d["reason"] == "thumbs up"
