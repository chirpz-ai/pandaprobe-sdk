"""Real embedded Bash: no fake shell or model required."""

import asyncio

import pytest

pytest.importorskip("bashkit")

from pandaprobe.reflect.config import ReflectConfig
from pandaprobe.reflect.models import Rule
from pandaprobe.reflect.shell import ReflectShell


def rule():
    return Rule(path="billing/refunds.md", text="# Refunds\nApplies to: refunds\nUse the authorized amount.\n")


def shell(**kwargs):
    return ReflectShell(ReflectConfig(**kwargs), {rule().path: rule()}, execution_id="execution")


async def test_ordinary_shell_operations_cover_rule_lifecycle():
    tool = shell()
    assert "authorized amount" in await tool.execute("grep -R -n authorized /rules")
    assert "billing/refunds.md" in await tool.execute("find /rules -name '*.md'")
    await tool.execute("sed -i 's/authorized/exact authorized/' /rules/billing/refunds.md")
    assert "exact authorized" in tool.rules()[rule().path].text
    await tool.execute("mkdir -p /rules/payments; mv /rules/billing/refunds.md /rules/payments/refunds.md")
    assert list(tool.rules()) == ["payments/refunds.md"]
    await tool.execute("rm /rules/payments/refunds.md")
    assert tool.rules() == {}
    await tool.execute(
        "cat > /rules/payments/new.md <<'EOF'\n# Payment\nApplies to: paying invoices\nVerify the payee.\nEOF"
    )
    assert tool.rules()["payments/new.md"].scope == "payments"
    await tool.aclose()


async def test_host_files_environment_and_network_are_unavailable(tmp_path, monkeypatch):
    secret = tmp_path / "host-secret.txt"
    secret.write_text("DO_NOT_EXPOSE_THIS")
    monkeypatch.setenv("PANDAPROBE_API_KEY", "HOST_API_KEY")
    tool = shell()
    result = await tool.execute(f"cat '{secret}'; echo $PANDAPROBE_API_KEY; curl https://example.com")
    assert "DO_NOT_EXPOSE_THIS" not in result and "HOST_API_KEY" not in result
    assert "[exit_code=0" not in result
    await tool.aclose()


async def test_evidence_and_decisions_remain_read_only_after_rollback():
    tool = shell(context_max_bytes=80)
    tool.set_evidence({"id": "notice"}, "original evidence")
    tool.set_decisions({"status": "pending"})
    await tool.execute(
        "echo modified > /inbox/evidence.txt; chmod 777 /decisions/status.json; echo bad > /decisions/status.json"
    )
    assert "original evidence" in await tool.execute("cat /inbox/evidence.txt")
    assert '"pending"' in await tool.execute("cat /decisions/status.json")
    await tool.execute("printf '%1000s' x > /context.md")
    await tool.execute("echo modified > /inbox/evidence.txt")
    assert "original evidence" in await tool.execute("cat /inbox/evidence.txt")
    await tool.aclose()


async def test_status_search_and_filename_do_not_count_as_full_delivery():
    tool = shell()
    await tool.execute("echo /rules/billing/refunds.md")
    await tool.execute("grep authorized /rules/billing/refunds.md")
    assert all(not receipt.rule_versions for receipt in tool.receipts)
    await tool.execute("cat /rules/billing/refunds.md")
    assert tool.receipts[-1].rule_versions == {rule().path: rule().version}
    assert not tool.receipts[-1].submitted  # Framework must confirm actual model submission.
    await tool.aclose()


async def test_context_edits_replace_the_file_and_respect_budget():
    tool = shell(context_max_bytes=80)
    await tool.execute("printf 'First note' > /context.md")
    assert tool.context() == "First note"
    await tool.execute("printf 'Replacement note' > /context.md")
    assert tool.context() == "Replacement note"
    result = await tool.execute("printf '%1000s' x > /context.md")
    assert "rolled back" in result
    assert tool.context() == "Replacement note"
    await tool.aclose()


async def test_command_file_and_output_limits():
    tool = shell(rule_max_bytes=100, output_max_bytes=512, command_max_bytes=256)
    assert "input budget" in await tool.execute("x" * 257)
    result = await tool.execute("printf '%1000s' x > /rules/large.md")
    assert "rolled back" in result and "large.md" not in tool.files()
    result = await tool.execute("printf '%10000s' x")
    assert len(result.encode()) <= 512 and "truncated" in result
    await tool.aclose()


async def test_repeated_guard_responses_cannot_exceed_the_output_allowance():
    tool = shell(command_max_bytes=16, execution_output_max_bytes=600, output_max_bytes=512)
    outputs = [await tool.execute("echo start")]
    outputs.extend([await tool.execute("x" * 17) for _ in range(100)])
    assert sum(len(output.encode("utf-8")) for output in outputs) <= 600
    assert outputs[-1] == "" and tool.usage["output_bytes"] == 600
    await tool.aclose()
    assert await tool.execute("echo late") == ""


async def test_reflection_has_an_independent_command_budget():
    tool = shell(reflection_max_commands=2)
    await tool.execute("echo task")
    tool.begin_reflection()
    await tool.execute("echo reflection1")
    await tool.execute("echo reflection2")
    assert "command budget reached" in await tool.execute("echo never")
    tool.end_reflection()
    assert "resumed" in await tool.execute("echo resumed")
    await tool.aclose()


async def test_persistent_state_and_instance_isolation():
    a, b = shell(), shell()
    await a.execute("export NOTE=hello; cd /rules")
    assert "hello" in await a.execute("echo $NOTE; pwd")
    assert "hello" not in await b.execute("echo $NOTE; pwd")
    await a.aclose()
    await b.aclose()


async def test_cancellation_does_not_leave_uncontrolled_shell_work():
    tool = shell(shell_timeout=2, shell_max_commands=100_000_000, shell_max_loop_iterations=100_000_000)
    task = asyncio.create_task(tool.execute("while true; do :; done"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "recovered" in await tool.execute("echo recovered")
    await tool.aclose()


async def test_invalid_paths_and_symlinks_never_export_outside_rules():
    tool = shell()
    result = await tool.execute("ln -s /inbox /rules/linked")
    assert "rolled back" in result
    assert list(tool.rules()) == [rule().path]
    await tool.execute("rm -rf /rules")
    assert tool.rules() == {}
    await tool.aclose()


async def test_checkpoints_persist_working_copy_and_receipts():
    changes, receipts = [], []
    tool = ReflectShell(
        ReflectConfig(),
        {},
        execution_id="e",
        on_checkpoint=lambda files, context: changes.append((files, context)),
        on_receipt=lambda receipt, text: receipts.append((receipt, text)),
    )
    await tool.execute("echo 'active notes' > /context.md")
    assert changes[-1][1] == "active notes\n"
    assert receipts[-1][0].byte_count == len(receipts[-1][1].encode())
    await tool.aclose()
