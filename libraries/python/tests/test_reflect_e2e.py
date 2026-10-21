"""Real HTTP, trace export, Bash, agent reflection, replay, and later delivery."""

import importlib.util
from pathlib import Path


async def test_full_native_loop_against_local_http_service(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "reflect" / "offline_demo.py"
    spec = importlib.util.spec_from_file_location("reflect_offline_demo", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = await module.demonstrate(tmp_path)
    assert report["original_result"] == 5
    assert report["candidate_status"] == "proposed"
    assert report["accepted_before_validation"] == []
    assert report["validation"] == "accepted" and report["later_result"] == 9
    assert report["accepted_files"] == ["sums.md"]
    assert report["trace_roles"] == ["reflection", "task", "validation"]
    assert report["evaluated_roles"] == ["task"]
    assert report["evaluation_creates"] == 3  # detect + diagnose, then later detect
    assert all(count >= 2 for count in report["poll_counts"])
    assert report["delayed_trace_reads"] >= 2
    assert report["real_model_calls"] == 0
