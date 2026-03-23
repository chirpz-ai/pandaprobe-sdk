"""LLM client wrappers for automatic span instrumentation.

Each provider lives in its own sub-package (e.g. ``wrappers/openai/``).
Adding a new provider only requires creating a new sub-package and
re-exporting the ``wrap_*`` function here.
"""

from pandaprobe.wrappers.openai import wrap_openai
from pandaprobe.wrappers.gemini import wrap_gemini
from pandaprobe.wrappers.anthropic import wrap_anthropic

__all__ = ["wrap_openai", "wrap_gemini", "wrap_anthropic"]
