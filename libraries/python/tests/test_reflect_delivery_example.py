"""Example retrieval evidence distinguishes verified excerpts from whole-file coverage."""

import json
import sys
from pathlib import Path

import pytest

from pandaprobe.reflect.models import Rule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/reflect"))
from _support.delivery import inspect_delivery, match_rules

RULE = Rule(path="billing/refunds.md", text="# Refunds\nApplies to: standard returns\n\nRefund merchandise only.\n")


def numbered(rule=RULE):
    return "\n".join(
        f"/rules/{rule.path}:{index}:{line}" for index, line in enumerate(rule.text.splitlines(), 1) if line.strip()
    )


def test_actual_live_grep_output_has_complete_verified_coverage():
    data = json.loads((Path(__file__).parent / "fixtures/reflect/numbered_grep.json").read_text())
    rule = Rule(path=data["path"], text=data["rule"])
    match = match_rules(data["output"], {rule.path: rule.text})[rule.path]
    assert match.complete and match.representation == "numbered_grep"
    assert match.matched_lines == [1, 2, 4] and match.total_nonblank_lines == 3
    assert match.version == rule.version


@pytest.mark.parametrize(
    "text", [RULE.text, numbered(), "prefix\n" + numbered() + "\n[exit_code=0]", "Observation: " + numbered()]
)
def test_whole_rule_coverage_preserves_version(text):
    match = match_rules(text, {RULE.path: RULE.text})[RULE.path]
    assert match.complete and match.version == RULE.version and match.matched_lines == [1, 2, 4]


@pytest.mark.parametrize(
    "text",
    [
        f"/rules/{RULE.path}",
        f"grep -n refund /rules/{RULE.path}",
        numbered().replace("/rules/billing/", "billing/"),
        numbered().replace("/rules/billing/", "/rules/other/"),
        numbered().replace("/rules/billing/", "/other/rules/billing/"),
        numbered().replace(":1:", ":99:").replace(":2:", ":0:").replace(":4:", ":-4:"),
        numbered().replace(":1:", ":" + "9" * 5000 + ":").replace(":2:", ":x:").replace(":4:", ":４:"),
    ],
)
def test_names_commands_wrong_paths_and_invalid_numbers_are_not_coverage(text):
    assert not match_rules(text, {RULE.path: RULE.text})


@pytest.mark.parametrize(
    "text",
    [
        numbered().splitlines()[-1],  # The actionable sentence alone omits applicability.
        "\n".join(numbered().splitlines()[:2]),  # Title/scope alone omit the behavior.
        numbered().replace("Refund merchandise only.", "Refund shipping too."),
        numbered().replace(":4:", ":3:"),
        numbered().replace("\n/rules/billing/refunds.md:4:", "\n--\n/rules/billing/refunds.md:4:"),
        numbered().replace("\n/rules/billing/refunds.md:4:", "\nother message\n/rules/billing/refunds.md:4:"),
        numbered().replace(":2:Applies to: standard returns", ":1:Applies to: standard returns"),
    ],
)
def test_partial_mismatched_or_disjoint_blocks_cannot_claim_full_delivery(text):
    match = match_rules(text, {RULE.path: RULE.text})[RULE.path]
    assert not match.complete and match.matched_lines


def test_rule_edits_do_not_relabel_old_output_as_delivery_of_new_version():
    edited = RULE.model_copy(update={"text": RULE.text.replace("merchandise only", "merchandise minus fees")})
    match = match_rules(numbered(), {edited.path: edited.text})[edited.path]
    assert match.version == edited.version and not match.complete
    assert match.matched_lines == [1, 2]


def test_numbered_text_is_matched_per_file_without_cross_rule_assembly():
    other = Rule(path="shipping.md", text="# Shipping\nApplies to: shipping refunds\nAsk for authorization.\n")
    matches = match_rules(numbered() + "\n" + numbered(other), {RULE.path: RULE.text, other.path: other.text})
    assert matches[RULE.path].complete and matches[other.path].complete
    assert matches[RULE.path].version != matches[other.path].version


@pytest.mark.parametrize(
    "started_at,before",
    [
        ("2026-10-23T00:00:00+00:00", True),
        ("2026-10-23T00:00:02+00:00", False),
        (None, False),
        ("invalid", False),
        ("2026-10-23T00:00:00", False),
    ],
)
def test_example_audit_uses_recorded_model_request_time(started_at, before):
    trace = {
        "trace_id": "task",
        "spans": [
            {"span_id": "model", "kind": "LLM", "started_at": started_at, "input": numbered()},
        ],
    }
    evidence = inspect_delivery([trace], {RULE.path: RULE.text}, "2026-10-23T00:00:01+00:00")
    assert evidence["model_input_versions"] == [RULE.version]
    assert evidence["before_submission_versions"] == ([RULE.version] if before else [])
    assert evidence["before_submission_excerpt_versions"] == ([RULE.version] if before else [])


@pytest.mark.parametrize("before", [False, True])
def test_verified_excerpt_counts_as_retrieval_only_before_submission(before):
    trace = {
        "trace_id": "task",
        "spans": [
            {
                "span_id": "model",
                "kind": "LLM",
                "started_at": f"2026-10-23T00:00:0{0 if before else 2}+00:00",
                "input": numbered().splitlines()[-1],
            }
        ],
    }
    evidence = inspect_delivery([trace], {RULE.path: RULE.text}, "2026-10-23T00:00:01+00:00")
    assert evidence["before_submission_excerpt_versions"] == ([RULE.version] if before else [])
    assert evidence["partial_model_input_versions"] == [RULE.version]
    assert not evidence["model_input_versions"] and not evidence["before_submission_versions"]
    coverage = evidence["receipts"][0]["rule_coverage"][RULE.path]
    assert coverage["matched_lines"] == [4] and not coverage["complete"]


def test_example_audit_does_not_treat_tool_output_as_model_input():
    trace = {
        "trace_id": "task",
        "spans": [{"span_id": "tool", "kind": "TOOL", "output": RULE.text}],
    }
    evidence = inspect_delivery([trace], {RULE.path: RULE.text}, None)
    assert not evidence["model_input_versions"] and not evidence["receipts"]
    assert not evidence["before_submission_excerpt_versions"]


async def test_example_matcher_distinguishes_partial_and_full_real_bash_output():
    pytest.importorskip("bashkit")
    from pandaprobe.reflect import ReflectConfig
    from pandaprobe.reflect.shell import ReflectShell

    shell = ReflectShell(ReflectConfig(), {RULE.path: RULE})
    try:
        partial = await shell.execute("grep -Rn merchandise /rules")
        assert not match_rules(partial, shell.files())[RULE.path].complete
        complete = await shell.execute("grep -RnE 'Refund|Applies' /rules")
        match = match_rules(complete, shell.files())[RULE.path]
        assert match.complete and match.version == RULE.version
    finally:
        await shell.aclose()


async def test_example_matcher_keeps_separate_excerpts_and_truncated_output_partial():
    pytest.importorskip("bashkit")
    from pandaprobe.reflect import ReflectConfig
    from pandaprobe.reflect.shell import ReflectShell

    rule = Rule(path="long.md", text="# Refunds\nApplies to: standard returns\n" + "Refund merchandise only. " * 40)
    shell = ReflectShell(ReflectConfig(output_max_bytes=512), {rule.path: rule})
    try:
        output = await shell.execute("grep -RnE 'Refund|Applies' /rules")
        assert len(output.encode()) <= 512
        assert not match_rules(output, shell.files())[rule.path].complete
        for line in numbered(rule).splitlines():
            assert not match_rules(line, shell.files())[rule.path].complete
    finally:
        await shell.aclose()
