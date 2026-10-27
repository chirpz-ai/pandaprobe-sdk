"""Regenerate deterministic fixtures from the released Python implementation.
Run from repo root: libraries/python/.venv/bin/python -B libraries/typescript/scripts/update-reflect-fixtures.py
"""
import hashlib
import json
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "libraries/python/src"))
from pandaprobe.reflect.models import Rule
from pandaprobe.reflect.workspace import _json
rules = [Rule(path="refunds/shipping.md", text="# Refund café 🐼\r\nApplies to: return refunds\r\nExclude outbound shipping.\r\n"), Rule(path="global.md", text="  # Safety\nApplies to: all tasks\nCheck evidence.  ")]
manifest = {rule.path: rule.version for rule in rules}
base = hashlib.sha256(_json({}).encode()).hexdigest()
changes = {rule.path: rule.model_dump() for rule in rules}
inputs = {"café": [True, None, 7, 0.25, "🐼"], "a": "line\nnext"}
fixture = {
    "rules": [{**rule.model_dump(), "version": rule.version, "scope": rule.scope, "applicability": rule.applicability} for rule in rules],
    "empty_revision": base, "manifest": manifest,
    "revision_json": _json(manifest), "revision": hashlib.sha256(_json(manifest).encode()).hexdigest(),
    "proposal_json": _json({"base": base, "changes": changes}),
    "proposal_fingerprint": hashlib.sha256(_json({"base": base, "changes": changes}).encode()).hexdigest(),
    "evaluation_json": json.dumps(["execution", "detect", ["trace"], ["task_completion"]], sort_keys=True),
    "input": inputs, "input_json": json.dumps(inputs, sort_keys=True),
    "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
}
(root / "libraries/typescript/test/fixtures/reflect-parity.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2)+"\n")

# Decision fixtures execute Python's actual assessor, excluding timestamps/UUIDs.
from copy import deepcopy
from types import SimpleNamespace
from pandaprobe.reflect.config import ReflectConfig
from pandaprobe.reflect.models import RuleProposal
from pandaprobe.reflect.validation import Validator
proposal = RuleProposal(id="fixture-proposal", base_revision=base, changes=changes,
                        fingerprint=fixture["proposal_fingerprint"], session_id="s", execution_id="e")
rows = [{"case_id": str(i), "base_revision": base, "input_hash": "fixture", "protected": i == 2,
         "control": {"execution_id": f"control-{i}", "scores": {"task_completion": 1 if i == 2 else 0}, "success": i == 2, "candidate_ids": [], "evidence": {}},
         "candidate": {"execution_id": f"candidate-{i}", "scores": {"task_completion": 1}, "success": True, "candidate_ids": [proposal.id], "evidence": {}}} for i in range(3)]
workspace = SimpleNamespace(rules=lambda revision: {}, candidate_rules=lambda proposal: {r.path: r for r in rules})
validator = Validator(SimpleNamespace(config=ReflectConfig(thresholds={"task_completion": .5}), workspace=workspace))
variants = {"improvement": rows, "too_few": rows[:2]}
for name in ["regression", "no_improvement", "reused_execution", "wrong_assignment", "invalid_boolean", "metric_mismatch", "unprotected", "wrong_baseline"]:
    values = deepcopy(rows)
    if name == "regression": values[2]["candidate"]["success"] = False
    if name == "no_improvement":
        for row in values: row["candidate"]["scores"] = deepcopy(row["control"]["scores"])
    if name == "reused_execution": values[0]["candidate"]["execution_id"] = values[0]["control"]["execution_id"]
    if name == "wrong_assignment": values[0]["candidate"]["candidate_ids"] = []
    if name == "invalid_boolean": values[0]["candidate"]["success"] = 1
    if name == "metric_mismatch": values[0]["candidate"]["scores"]["extra"] = 1
    if name == "unprotected":
        for row in values: row["protected"] = False
    if name == "wrong_baseline": values[0]["base_revision"] = "x" * 64
    variants[name] = values
fixture["decisions"] = [{"name": name, "rows": values, **{k:v for k,v in validator.assess(proposal, values, mode="replay").model_dump().items() if k in ("outcome", "reason")}} for name, values in variants.items()]
fixture["decision_proposal"] = proposal.model_dump(exclude={"created_at"})
(root / "libraries/typescript/test/fixtures/reflect-parity.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2)+"\n")
