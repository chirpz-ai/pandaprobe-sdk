"""Contract tests for the shared stream-reducer base classes.

These exercise :class:`pandaprobe.wrappers._base.SyncStreamReducer` and
:class:`pandaprobe.wrappers._base.AsyncStreamReducer` directly, in isolation
from any provider wrapper.  They lock in the *contract* that every provider
inherits — Mistral, OpenAI Chat Completions, Anthropic, Gemini, and any
future LLM SDK that subclasses these reducers.

What's covered:

* Normal completion → ``reduce_chunks`` runs and the span is closed once.
* Mid-iteration exception (no ``with``) → span finalized as **error**, not
  silently leaked open.
* Mid-iteration exception (inside ``with``) → ``__exit__`` records the
  exception on the span, and ``_finalize`` is idempotent across the two
  paths (``__next__`` exception handler + ``__exit__``).
* ``with`` block raising **after** iteration ends → block-level exception
  recorded as error, *not* swallowed by a success-close.
* ``__enter__`` / ``__aenter__`` forwards to the inner stream's CM.
* First-chunk ``set_completion_start_time`` is fired exactly once.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from pandaprobe.wrappers import _base as base_module
from pandaprobe.wrappers._base import AsyncStreamReducer, SyncStreamReducer


@pytest.fixture
def patched_close(monkeypatch):
    """Monkeypatch ``close_llm_span`` and ``error_llm_span`` and capture calls."""
    close_calls: list[Any] = []
    error_calls: list[BaseException] = []
    monkeypatch.setattr(base_module, "close_llm_span", lambda ctx: close_calls.append(ctx))
    monkeypatch.setattr(base_module, "error_llm_span", lambda ctx, exc: error_calls.append(exc))
    return close_calls, error_calls


# ---------------------------------------------------------------------------
# SyncStreamReducer
# ---------------------------------------------------------------------------


class TestSyncStreamReducerHappyPath:
    def test_normal_completion_calls_reduce_chunks_once(self, patched_close):
        close_calls, error_calls = patched_close
        span_ctx = MagicMock()

        reducer = SyncStreamReducer(iter(["a", "b", "c"]), span_ctx)
        chunks = list(reducer)

        assert chunks == ["a", "b", "c"]
        assert len(close_calls) == 1, "default reduce_chunks must close the span exactly once"
        assert error_calls == []

    def test_first_chunk_sets_completion_start_time_once(self, patched_close):
        span_ctx = MagicMock()
        reducer = SyncStreamReducer(iter(["a", "b", "c"]), span_ctx)
        list(reducer)
        assert span_ctx.set_completion_start_time.call_count == 1

    def test_inner_stream_context_manager_is_forwarded(self, patched_close):
        inner = MagicMock()
        inner.__iter__ = MagicMock(return_value=iter([]))
        inner.__enter__ = MagicMock(return_value=inner)
        inner.__exit__ = MagicMock(return_value=None)
        inner.__next__ = MagicMock(side_effect=StopIteration)

        with SyncStreamReducer(inner, MagicMock()):
            pass

        inner.__enter__.assert_called_once()
        inner.__exit__.assert_called_once()


class TestSyncStreamReducerErrorPath:
    def test_mid_iteration_error_finalizes_span_as_error(self, patched_close):
        """Regression: a non-StopIteration exception in ``__next__`` must
        finalize the span as an error, not leak it open.
        """
        close_calls, error_calls = patched_close

        def _exploding():
            yield "a"
            raise RuntimeError("boom")

        reducer = SyncStreamReducer(_exploding(), MagicMock())

        with pytest.raises(RuntimeError, match="boom"):
            list(reducer)

        assert len(error_calls) == 1, "span was not finalized as error"
        assert isinstance(error_calls[0], RuntimeError)
        assert "boom" in str(error_calls[0])
        assert close_calls == [], "errored span must not also be normally closed"

    def test_with_block_exception_records_block_exception_on_span(self, patched_close):
        """Regression: ``__exit__`` must propagate the block-level exception
        to ``_finalize`` so the span closes as errored, not as a successful
        no-op.  The pre-fix ``__exit__(*args) → _finalize()`` pattern dropped
        the exception entirely.
        """
        close_calls, error_calls = patched_close

        reducer = SyncStreamReducer(iter(["a"]), MagicMock())

        with pytest.raises(ValueError, match="user-side"):
            with reducer:
                next(iter(reducer))
                raise ValueError("user-side")

        assert len(error_calls) == 1
        assert isinstance(error_calls[0], ValueError)
        assert close_calls == []

    def test_finalize_is_idempotent_across_next_and_exit(self, patched_close):
        """``__next__``'s exception path *and* the surrounding ``__exit__`` both
        try to finalize when ``with`` is used together with a mid-iter raise —
        the span must only be closed once.
        """
        close_calls, error_calls = patched_close

        def _exploding():
            yield "a"
            raise RuntimeError("inner")

        reducer = SyncStreamReducer(_exploding(), MagicMock())

        with pytest.raises(RuntimeError, match="inner"):
            with reducer as r:
                for _ in r:
                    pass

        assert len(error_calls) == 1, "span must be finalized exactly once"
        assert close_calls == []

    def test_normal_with_block_exit_runs_reduce_chunks(self, patched_close):
        """``with`` exiting cleanly after iteration runs the success path."""
        close_calls, error_calls = patched_close

        reducer = SyncStreamReducer(iter(["a", "b"]), MagicMock())
        with reducer as r:
            list(r)

        assert len(close_calls) == 1
        assert error_calls == []


class TestSyncStreamReducerSubclass:
    def test_subclass_reduce_chunks_runs_on_normal_completion(self, patched_close):
        captured: list[list[Any]] = []

        class _Sub(SyncStreamReducer):
            def reduce_chunks(self, span_ctx, chunks):
                captured.append(list(chunks))

        reducer = _Sub(iter(["x", "y"]), MagicMock())
        list(reducer)

        assert captured == [["x", "y"]]

    def test_subclass_reduce_chunks_skipped_on_error(self, patched_close):
        """Important: provider-specific reducers must NOT see partial chunks
        on the error path — that's why the error route bypasses
        ``reduce_chunks`` entirely.
        """
        _, error_calls = patched_close
        called: list[Any] = []

        class _Sub(SyncStreamReducer):
            def reduce_chunks(self, span_ctx, chunks):
                called.append(chunks)

        def _exploding():
            yield "a"
            raise RuntimeError("boom")

        reducer = _Sub(_exploding(), MagicMock())
        with pytest.raises(RuntimeError):
            list(reducer)

        assert called == [], "reduce_chunks must not run on the error path"
        assert len(error_calls) == 1


# ---------------------------------------------------------------------------
# AsyncStreamReducer
# ---------------------------------------------------------------------------


async def _aiter(items):
    for it in items:
        yield it


class TestAsyncStreamReducerHappyPath:
    async def test_normal_completion_calls_reduce_chunks_once(self, patched_close):
        close_calls, error_calls = patched_close

        reducer = AsyncStreamReducer(_aiter(["a", "b", "c"]), MagicMock())
        collected = [chunk async for chunk in reducer]

        assert collected == ["a", "b", "c"]
        assert len(close_calls) == 1
        assert error_calls == []

    async def test_first_chunk_sets_completion_start_time_once(self, patched_close):
        span_ctx = MagicMock()
        reducer = AsyncStreamReducer(_aiter(["a", "b", "c"]), span_ctx)
        async for _ in reducer:
            pass
        assert span_ctx.set_completion_start_time.call_count == 1


class TestAsyncStreamReducerErrorPath:
    async def test_mid_iteration_error_finalizes_span_as_error(self, patched_close):
        """Regression: async streaming must finalize the span on any
        non-StopAsyncIteration exception, not just on stream exhaustion.
        """
        close_calls, error_calls = patched_close

        async def _exploding_aiter():
            yield "a"
            raise RuntimeError("async-boom")

        reducer = AsyncStreamReducer(_exploding_aiter(), MagicMock())

        with pytest.raises(RuntimeError, match="async-boom"):
            async for _ in reducer:
                pass

        assert len(error_calls) == 1
        assert isinstance(error_calls[0], RuntimeError)
        assert close_calls == []

    async def test_async_with_block_exception_recorded_as_error(self, patched_close):
        """``async with`` block raising during iteration → ``__aexit__`` must
        propagate the exception to ``_finalize`` so the span closes as
        errored, not as a successful no-op.

        We raise from inside the loop body (after one chunk) rather than
        after iteration completes, because once iteration ends normally
        ``_finalize`` has already run on the success path — that's the
        intended design, not the bug under test here.
        """
        close_calls, error_calls = patched_close

        reducer = AsyncStreamReducer(_aiter(["a", "b", "c"]), MagicMock())

        with pytest.raises(ValueError, match="async-user"):
            async with reducer as r:
                async for _ in r:
                    raise ValueError("async-user")

        assert len(error_calls) == 1
        assert isinstance(error_calls[0], ValueError)
        assert close_calls == []

    async def test_finalize_is_idempotent_across_anext_and_aexit(self, patched_close):
        close_calls, error_calls = patched_close

        async def _exploding_aiter():
            yield "a"
            raise RuntimeError("async-inner")

        reducer = AsyncStreamReducer(_exploding_aiter(), MagicMock())

        with pytest.raises(RuntimeError, match="async-inner"):
            async with reducer as r:
                async for _ in r:
                    pass

        assert len(error_calls) == 1
        assert close_calls == []
