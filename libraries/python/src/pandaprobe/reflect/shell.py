"""The supplied, isolated Bash interface to Reflect's rule filesystem.

Bashkit executes shell syntax and built-in file utilities in a virtual
filesystem. No host directories, credentials, network, or external executables
are exposed. Only SDK code exports a bounded working revision to durable storage.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

from .config import ReflectConfig
from .models import DeliveryReceipt, Rule, digest
from .workspace import WorkspaceError

TOOL_NAME = "reflect_bash"
TOOL_DESCRIPTION = (
    "Run Bash commands in the persistent Reflect filesystem. Search/read /rules for learned guidance; "
    "edit Markdown rule files to propose improvements and /context.md to maintain active memory. "
    "Read /inbox for assigned failure evidence and /decisions for validation results. "
    "Supports grep, find, sed, awk, cat, mkdir, mv, rm, pipes, loops and heredocs. "
    "This filesystem is isolated from host files and network."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {"command": {"type": "string", "description": "Bash command to execute"}},
    "required": ["command"],
    "additionalProperties": False,
}


def clip_bytes(text: str, maximum: int) -> str:
    return text.encode("utf-8")[: max(0, maximum)].decode("utf-8", errors="ignore")


class ReflectShell:
    """One execution's persistent shell. Instances never share interpreter state."""

    def __init__(
        self,
        config: ReflectConfig,
        rules: Mapping[str, Rule],
        *,
        execution_id: str,
        context: str = "",
        candidate_ids: list[str] | None = None,
        on_receipt: Callable[[DeliveryReceipt, str], None] | None = None,
        on_checkpoint: Callable[[dict[str, str], str], None] | None = None,
        usage: Mapping[str, int] | None = None,
    ) -> None:
        try:
            from bashkit import Bash
        except ImportError as exc:
            raise ImportError('Install Reflect with pip install "pandaprobe[reflect]"') from exc
        self.config = config
        self.execution_id = execution_id
        self.candidate_ids = list(candidate_ids or [])
        self._on_receipt = on_receipt
        self._on_checkpoint = on_checkpoint
        self._lock = asyncio.Lock()
        self._output_bytes = (usage or {}).get("output_bytes", 0)
        self._commands = (usage or {}).get("commands", 0)
        self._command_limit: int | None = None
        self._closed = False
        self._bash = Bash(
            cwd="/",
            env={"PANDAPROBE_REFLECT": "1"},
            max_commands=config.shell_max_commands,
            max_loop_iterations=config.shell_max_loop_iterations,
            max_memory=config.shell_memory_bytes,
            timeout_seconds=config.shell_timeout,
            python=False,
            network=None,
            mounts=[],
        )
        self._bash.mkdir("/rules", recursive=True)
        for path, rule in rules.items():
            self._bash.mkdir(("/rules/" + path).rsplit("/", 1)[0], recursive=True)
            self._bash.write_file(f"/rules/{path}", rule.text)
        self._bash.write_file("/context.md", clip_bytes(context, config.context_max_bytes))
        self._mount_readonly("/inbox", {})
        self._mount_readonly("/decisions", {})
        self.receipts: list[DeliveryReceipt] = []

    def _mount_readonly(self, path: str, contents: Mapping[str, str]) -> None:
        from bashkit import FileSystem

        files = FileSystem()
        for name, content in contents.items():
            if "/" in name or name in (".", ".."):
                raise ValueError("Evidence and decision filenames must be simple names")
            files.write_file("/" + name, content.encode("utf-8"))
        self._bash.mount(path, files, read_only=True)

    def set_evidence(self, notice: Mapping[str, Any], evidence: str) -> None:
        """SDK-only input publication. Original evidence remains outside Bash."""
        self._bash.unmount("/inbox")
        self._mount_readonly(
            "/inbox",
            {
                "notice.json": json.dumps(dict(notice), ensure_ascii=False, indent=2),
                "evidence.txt": clip_bytes(evidence, self.config.evidence_max_bytes),
            },
        )

    def set_decisions(self, decisions: Mapping[str, Any]) -> None:
        self._bash.unmount("/decisions")
        self._mount_readonly("/decisions", {"status.json": json.dumps(dict(decisions), ensure_ascii=False)})

    def begin_reflection(self) -> None:
        self._command_limit = self._commands + self.config.reflection_max_commands

    def end_reflection(self) -> None:
        self._command_limit = None

    @property
    def command_count(self) -> int:
        return self._commands

    @property
    def usage(self) -> dict[str, int]:
        return {"commands": self._commands, "output_bytes": self._output_bytes}

    def context(self) -> str:
        if not self._bash.exists("/context.md"):
            return ""
        metadata = self._bash.stat("/context.md")
        if metadata["file_type"] != "file" or metadata["size"] > self.config.context_max_bytes:
            raise WorkspaceError("context.md must be a regular file within its byte budget")
        return self._bash.read_file("/context.md")

    def files(self) -> dict[str, str]:
        """Bounded, non-symlink traversal of the agent's working rule directory."""
        files: dict[str, str] = {}
        if not self._bash.exists("/rules"):
            return files
        if self._bash.stat("/rules")["file_type"] != "directory":
            raise WorkspaceError("/rules must be a directory")
        pending = ["/rules"]
        entries_seen, size = 0, 0
        while pending:
            directory = pending.pop()
            for entry in self._bash.read_dir(directory):
                entries_seen += 1
                if entries_seen > self.config.rules_max_count * 4 + 32:
                    raise WorkspaceError("Too many rule directory entries; consolidate the working revision")
                path = directory + "/" + entry["name"]
                info = entry["metadata"]
                if info["file_type"] == "directory":
                    if path.count("/") > 12:
                        raise WorkspaceError("Rule directory nesting exceeds the limit")
                    pending.append(path)
                elif info["file_type"] != "file":
                    raise WorkspaceError("Rule files cannot be symbolic links")
                elif path.endswith(".md"):
                    name = path.removeprefix("/rules/")
                    Rule.safe_path(name)
                    if info["size"] > self.config.rule_max_bytes:
                        raise WorkspaceError(f"Rule {name} exceeds its byte budget")
                    size += info["size"]
                    if size > self.config.rules_max_bytes or len(files) >= self.config.rules_max_count:
                        raise WorkspaceError("Rule corpus exceeds its file or byte budget")
                    files[name] = self._bash.read_file(path)
        return files

    def rules(self) -> dict[str, Rule]:
        return {path: Rule(path=path, text=text) for path, text in self.files().items()}

    def receipt(self, text: str, *, channel: str, submitted: bool = False) -> DeliveryReceipt:
        # Only full exact rule text earns a body-delivery receipt. grep snippets,
        # filenames, status lookups, and commands mentioning a file do not.
        versions: dict[str, str] = {}
        for path, content in self.files().items():
            if content.strip() and content.strip() in text:
                versions[path] = digest(content.replace("\r\n", "\n").strip() + "\n")
        receipt = DeliveryReceipt(
            execution_id=self.execution_id,
            channel=channel,
            content_hash=digest(text),
            byte_count=len(text.encode("utf-8")),
            rule_versions=versions,
            candidate_ids=list(self.candidate_ids),
            submitted=submitted,
        )
        self.receipts.append(receipt)
        if self._on_receipt:
            self._on_receipt(receipt, text)
        return receipt

    def _control_message(self, message: str) -> str:
        remaining = self.config.execution_output_max_bytes - self._output_bytes
        rendered = clip_bytes(message, min(self.config.output_max_bytes, remaining))
        if rendered:
            self._output_bytes += len(rendered.encode("utf-8"))
            if self._on_checkpoint:
                self._on_checkpoint(self.files(), self.context())
            self.receipt(rendered, channel="tool_output")
        return rendered

    async def execute(self, command: str) -> str:
        async with self._lock:
            if self._closed:
                return ""  # Late callers cannot add output after this boundary has closed.
            if not isinstance(command, str) or len(command.encode("utf-8")) > self.config.command_max_bytes:
                return self._control_message("Command exceeds the Reflect input budget.")
            if self._command_limit is not None and self._commands >= self._command_limit:
                return self._control_message(
                    "Reflection command budget reached. Return to the task; unfinished guidance remains pending."
                )
            remaining = self.config.execution_output_max_bytes - self._output_bytes
            if remaining < 256:
                return self._control_message("Reflect output budget reached for this execution.")
            snapshot = self._bash.snapshot()
            self._bash.clear_cancel()
            self._commands += 1

            async def run() -> Any:
                return await self._bash.execute(command)

            task = asyncio.create_task(run())
            try:
                result = await asyncio.wait_for(asyncio.shield(task), self.config.shell_timeout + 1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._bash.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), self.config.shell_timeout + 1)
                except (Exception, asyncio.CancelledError):  # noqa: BLE001 - preserve the caller's cancellation
                    task.cancel()
                self._bash.restore_snapshot(snapshot)
                raise
            except Exception:  # noqa: BLE001 - the native interpreter reports errors across an FFI boundary
                self._bash.restore_snapshot(snapshot)
                return self._control_message("Reflect command failed; its filesystem changes were rolled back.")
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            code = result.exit_code
            try:
                files, context = self.files(), self.context()
            except (ValueError, WorkspaceError) as exc:
                self._bash.restore_snapshot(snapshot)
                output = f"Filesystem change rejected and rolled back: {exc}"
                code = 1
                files, context = self.files(), self.context()
            limit = min(self.config.output_max_bytes, remaining)
            truncated = len(output.encode("utf-8")) > limit - 256 or result.stdout_truncated or result.stderr_truncated
            body = clip_bytes(output, limit - 256)
            suffix = (
                f"\n[exit_code={code}; context.md={len(context.encode('utf-8'))}/"
                f"{self.config.context_max_bytes} bytes"
                + ("; output truncated: use a narrower search or excerpt" if truncated else "")
                + "]"
            )
            rendered = clip_bytes(body + suffix, limit)
            self._output_bytes += len(rendered.encode("utf-8"))
            if self._on_checkpoint:
                self._on_checkpoint(files, context)
            self.receipt(rendered, channel="tool_output")
            return rendered

    async def aclose(self) -> None:
        self._closed = True
        self._bash.cancel()
        async with self._lock:
            pass

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS},
        }
