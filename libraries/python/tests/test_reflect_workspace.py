"""Filesystem authority survives crashes, competing instances, and agent edits."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from pandaprobe.reflect.config import ReflectConfig
from pandaprobe.reflect.models import Rule, ValidationVerdict
from pandaprobe.reflect.workspace import RevisionConflict, Workspace, WorkspaceError


def rule(path="billing/refunds.md", guidance="Refund exactly the authorized amount."):
    return Rule(path=path, text=f"# Refunds\nApplies to: issuing refunds\n{guidance}\n")


def propose(store, rules=None):
    return store.propose(
        store.state()["head"],
        rules or {"billing/refunds.md": rule()},
        session_id="session",
        execution_id="execution",
        evidence_ids=["failure"],
    )


def verdict(proposal, outcome="accepted"):
    return ValidationVerdict(
        proposal_id=proposal.id, outcome=outcome, reason="Independent test outcome", mode="replay"
    )


def test_candidates_are_not_published_until_independent_decision(tmp_path):
    store = Workspace(tmp_path, ReflectConfig())
    initial = store.state()["head"]
    result = propose(store)
    assert result.status == "proposed"
    assert store.rules() == {} and store.state()["head"] == initial
    assert store.candidate_rules(result.proposal) == {"billing/refunds.md": rule()}
    store.decide(verdict(result.proposal))
    assert store.rules() == {"billing/refunds.md": rule()}
    assert store.rules(initial) == {}
    restarted = Workspace(tmp_path, ReflectConfig())
    assert restarted.proposal(result.proposal.id).status == "accepted"
    assert restarted.rules() == store.rules()


def test_missing_authoritative_state_cannot_reset_nonempty_history(tmp_path):
    store = Workspace(tmp_path, ReflectConfig())
    store.decide(verdict(propose(store).proposal))
    (tmp_path / "state.json").unlink()
    with pytest.raises(WorkspaceError, match="state is missing"):
        Workspace(tmp_path, ReflectConfig())
    assert not (tmp_path / "state.json").exists()


def test_edits_merges_moves_and_deletions_are_changesets(tmp_path):
    store = Workspace(tmp_path, ReflectConfig())
    original = propose(store).proposal
    store.decide(verdict(original))
    new = rule("payments/refunds.md", "Use the authorized amount and record the refund ID.")
    candidate = propose(store, {new.path: new}).proposal
    assert candidate.changes["billing/refunds.md"] is None
    assert candidate.changes[new.path] == new
    store.decide(verdict(candidate, "rejected"))
    assert store.rules() == {"billing/refunds.md": rule()}
    deletion = store.propose(store.state()["head"], {}, session_id="s", execution_id="e").proposal
    assert deletion.changes == {"billing/refunds.md": None}
    store.decide(verdict(deletion))
    assert store.rules() == {}


def test_stale_proposal_cannot_overwrite_concurrent_acceptance(tmp_path):
    first, second = Workspace(tmp_path, ReflectConfig()), Workspace(tmp_path, ReflectConfig())
    a = propose(first).proposal
    b_rule = rule(guidance="Different behavior.")
    b = propose(second, {b_rule.path: b_rule}).proposal
    first.decide(verdict(a))
    with pytest.raises(RevisionConflict):
        second.decide(verdict(b))
    assert second.proposal(b.id).status == "candidate"
    assert second.rules() == first.rules()
    assert second.propose(b.base_revision, {}, session_id="s", execution_id="e").status == "no_proposal"
    assert (
        second.propose(b.base_revision, {b_rule.path: b_rule}, session_id="s", execution_id="e").status == "conflict"
    )


def test_duplicate_and_no_change_outcomes(tmp_path):
    store = Workspace(tmp_path, ReflectConfig())
    first = propose(store)
    second = propose(store)
    assert second.status == "duplicate" and second.proposal.id == first.proposal.id
    assert store.propose(store.state()["head"], {}, session_id="s", execution_id="e").status == "no_proposal"
    duplicate = rule("billing/copy.md")
    assert propose(store, {"billing/refunds.md": rule(), duplicate.path: duplicate}).status == "duplicate"


def test_atomic_head_publication_survives_interruption(tmp_path, monkeypatch):
    import pandaprobe.reflect.workspace as module

    store = Workspace(tmp_path, ReflectConfig())
    proposal = propose(store).proposal
    initial = store.state()["head"]
    replace = module.os.replace

    def interrupted(src, dest):
        if Path(dest).name == "state.json":
            raise OSError("simulated crash before head publication")
        return replace(src, dest)

    monkeypatch.setattr(module.os, "replace", interrupted)
    with pytest.raises(OSError):
        store.decide(verdict(proposal))
    assert store.state()["head"] == initial
    assert store.rules() == {}
    monkeypatch.setattr(module.os, "replace", replace)
    restarted = Workspace(tmp_path, ReflectConfig())
    restarted.decide(verdict(proposal))
    assert restarted.rules() == {"billing/refunds.md": rule()}


@pytest.mark.parametrize("path", ["../escape.md", "/absolute.md", "a/../../b.md", "a\\b.md", "a.json", "x/./a.md"])
def test_rule_paths_cannot_escape(path):
    with pytest.raises(ValidationError):
        rule(path)


def test_rule_metadata_and_size_limits(tmp_path):
    with pytest.raises(ValidationError):
        Rule(path="a.md", text="unscoped instructions")
    store = Workspace(tmp_path, ReflectConfig(rule_max_bytes=20))
    with pytest.raises(WorkspaceError, match="byte budget"):
        propose(store)


def test_revision_tampering_is_detected(tmp_path):
    store = Workspace(tmp_path, ReflectConfig())
    store.decide(verdict(propose(store).proposal))
    file = tmp_path / "revisions" / store.state()["head"] / "rules/billing/refunds.md"
    file.write_text(rule(guidance="Changed without validation.").text)
    with pytest.raises(WorkspaceError, match="integrity"):
        store.rules()


def test_instances_do_not_lose_concurrent_state_updates(tmp_path):
    def update(_):
        store = Workspace(tmp_path, ReflectConfig())
        store.transaction(lambda state: state.update(evaluation_count=state["evaluation_count"] + 1))

    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(update, range(24)))
    assert Workspace(tmp_path, ReflectConfig()).state()["evaluation_count"] == 24


def _process_updates(path):
    store = Workspace(Path(path), ReflectConfig())
    for _ in range(5):
        store.transaction(lambda state: state.update(evaluation_count=state["evaluation_count"] + 1))


def test_cross_process_locking(tmp_path):
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_process_updates, args=(str(tmp_path),)) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert Workspace(tmp_path, ReflectConfig()).state()["evaluation_count"] == 15


def test_execution_ids_are_not_paths(tmp_path):
    store = Workspace(tmp_path, ReflectConfig())
    store.save_execution("../../elsewhere", {"pending": True})
    assert store.load_execution("../../elsewhere") == {"pending": True}
    assert store.load_execution("other") is None
