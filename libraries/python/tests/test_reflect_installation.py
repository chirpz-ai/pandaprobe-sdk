"""Imports must remain inert even when every optional dependency is unavailable."""

import os
import subprocess
import sys


def test_minimal_imports_do_not_activate_reflect_or_load_optional_packages(tmp_path):
    source = r"""
import importlib.abc
import pathlib
import sys

# Installed namespace packages may already be registered by site-packages .pth files.
initial_modules = set(sys.modules)
blocked = {"bashkit", "filelock", "agents", "langchain", "langchain_core", "langgraph", "crewai", "claude_agent_sdk", "google"}
class NoOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ImportError("Optional package must not load: " + fullname)
sys.meta_path.insert(0, NoOptionalPackages())
import pandaprobe
assert "pandaprobe.reflect" not in sys.modules
from pandaprobe.api import PlatformClient
from pandaprobe.reflect import Reflect, ReflectConfig
assert importlib.util.find_spec("pandaprobe.reflect.delivery") is None
assert "_support.delivery" not in sys.modules
client = pandaprobe.Client(enabled=False)
with client.trace("disabled"):
    pass
client.shutdown()
assert not pathlib.Path(".pandaprobe").exists()
assert not any(name.split(".")[0] in blocked for name in set(sys.modules) - initial_modules)
"""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PANDAPROBE_")}
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
