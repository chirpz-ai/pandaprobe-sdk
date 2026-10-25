"""Example-only inspection of rule text in recorded task-model inputs.

No shell-command interpretation or semantic similarity: a filename, matching
phrase, or model claim cannot establish delivery of a whole rule version.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime

from pandaprobe.reflect.integrations import input_text
from pandaprobe.reflect.models import digest, utc_now


@dataclass(frozen=True)
class RuleCoverage:
    version: str
    representation: str
    matched_lines: list[int]
    total_nonblank_lines: int
    complete: bool


class TaskInputAudit:
    """Example-owned observation where native async tracing omits LLM inputs.

    Attached only to this example's LangChain-family invocation. It records task
    input, never changes prompts/tools or participates in SDK validation.
    """

    def __init__(self):
        self.execution_id = None
        self.records = []

    def middleware(self):
        from langchain.agents.middleware import AgentMiddleware

        audit = self

        class Middleware(AgentMiddleware):
            async def awrap_model_call(self, request, handler):
                audit.record([request.system_message, request.messages], None)
                return await handler(request)

        return Middleware()

    def attach(self, binding):
        if not hasattr(binding.agent, "ainvoke"):
            return
        from langchain_core.callbacks import AsyncCallbackHandler

        audit = self

        class Callback(AsyncCallbackHandler):
            async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
                audit.record(messages, run_id)

            async def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
                audit.record(prompts, run_id)

        config = dict(binding.kwargs.get("config") or {})
        callbacks = config.get("callbacks")
        if callbacks is not None and not isinstance(callbacks, list):
            callbacks = callbacks.copy()
            callbacks.add_handler(Callback())
        else:
            callbacks = [*(callbacks or []), Callback()]
        binding.kwargs["config"] = {**config, "callbacks": callbacks}

    def record(self, value, run_id):
        if self.execution_id is not None:
            self.records.append(
                {
                    "span_id": str(run_id) if run_id is not None else None,
                    "kind": "LLM",
                    "started_at": utc_now(),
                    "input": input_text(value),
                    "execution_id": self.execution_id,
                }
            )


def _numbered_lines(text: str, path: str, source: list[str], required: set[int]) -> set[int]:
    """Best contiguous block of verified /rules/path:line:text grep records.

    Do not stitch together separate excerpts, versions, or history entries. A gap
    in numbering is fine only when the missing source lines are blank. Unnumbered
    grep, relative/ambiguous paths and grep context separators remain conservative.
    """
    prefix = f"/rules/{path}:"
    best: set[int] = set()
    block: set[int] = set()
    previous = 0
    for rendered in text.split("\n"):
        valid = False
        offset = rendered.find(prefix)
        # Frameworks such as CrewAI prefix tool results with "Observation: ".
        # A whitespace boundary permits that wrapper, never a suffix of another path.
        if offset == 0 or (offset > 0 and rendered[offset - 1].isspace()):
            number, separator, value = rendered[offset + len(prefix) :].partition(":")
            # Bound digit parsing even when arbitrary model input contains a huge integer.
            if separator and number.isascii() and number.isdecimal() and len(number) <= 9:
                index = int(number)
                valid = 1 <= index <= len(source) and source[index - 1] == value
                if valid:
                    if index <= previous:
                        block = set()
                    previous = index
                    if index in required:
                        block.add(index)
                    if len(block) > len(best):
                        best = set(block)
        if not valid:
            block, previous = set(), 0
    return best


def match_rules(text: str, files: Mapping[str, str]) -> dict[str, RuleCoverage]:
    """Report exact full text or verified numbered lines, keeping partials explicit."""
    coverage = {}
    for path, content in files.items():
        source = content.replace("\r\n", "\n")
        required = {index for index, line in enumerate(source.split("\n"), 1) if line.strip()}
        if not required:
            continue
        if source.strip() in text:
            matched, representation = required, "verbatim"
        else:
            matched = _numbered_lines(text, path, source.split("\n"), required)
            representation = "numbered_grep"
        if matched:
            coverage[path] = RuleCoverage(
                version=digest(source.strip() + "\n"),
                representation=representation,
                matched_lines=sorted(matched),
                total_nonblank_lines=len(required),
                complete=matched == required,
            )
    return coverage


def inspect_delivery(traces: list[dict], files: Mapping[str, str], first_submit: str | None) -> dict:
    """Example assertion over existing LLM traces, never a Reflect acceptance policy.

    Use recorded request timestamps, not the later time this audit runs. Callers
    supply only task trace IDs so subsequent reflection cannot satisfy the check.
    """
    receipts = []
    for trace in traces:
        for span in trace.get("spans", []):
            if span.get("kind") != "LLM" or span.get("input") is None:
                continue
            coverage = match_rules(input_text(span["input"]), files)
            started_at = span.get("started_at")
            before = False
            if started_at and first_submit:
                try:
                    before = datetime.fromisoformat(started_at) < datetime.fromisoformat(first_submit)
                except (TypeError, ValueError):
                    pass  # Missing/ambiguous timing cannot demonstrate a before-action read.
            receipts.append(
                {
                    "trace_id": trace["trace_id"],
                    "source": trace.get("source", "recorded task trace"),
                    "execution_id": span.get("execution_id"),
                    "span_id": span["span_id"],
                    "started_at": started_at,
                    "before_submission": before,
                    "rule_versions": {path: row.version for path, row in coverage.items() if row.complete},
                    "rule_coverage": {path: asdict(row) for path, row in coverage.items()},
                }
            )
    full = {version for row in receipts for version in row["rule_versions"].values()}
    partial = {
        match["version"] for row in receipts for match in row["rule_coverage"].values() if not match["complete"]
    }
    return {
        "source": "example audit of task LLM inputs",
        "receipts": receipts,
        "model_input_versions": sorted(full),
        "partial_model_input_versions": sorted(partial - full),
        "before_submission_versions": sorted(
            {version for row in receipts if row["before_submission"] for version in row["rule_versions"].values()}
        ),
    }


async def collect_delivery(runtime, trace_ids, files, events, *, observed_inputs=()):
    """Read existing task traces with a bounded deadline; never create evaluations."""
    first_submit = next((event["attempted_at"] for event in events if event["operation"] == "submit"), None)
    observed = [{"trace_id": None, "source": "example native observation", "spans": list(observed_inputs)}]
    empty = inspect_delivery(observed, files, first_submit)
    if not files or not trace_ids:
        return empty
    try:
        await runtime.wait_export(timeout=10)
        traces = await asyncio.wait_for(
            asyncio.gather(
                *(runtime.api.traces.wait_visible(trace_id, timeout=30) for trace_id in dict.fromkeys(trace_ids))
            ),
            timeout=30,
        )
        return inspect_delivery([*observed, *(trace.model_dump(mode="json") for trace in traces)], files, first_submit)
    except Exception as exc:
        return {**empty, "error": type(exc).__name__}
