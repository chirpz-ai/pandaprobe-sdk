"""LLM client wrappers for automatic span instrumentation.

Each provider lives in its own sub-package (e.g. ``wrappers/openai/``).
Adding a new provider only requires creating a new sub-package and
re-exporting the ``wrap_*`` function here.
"""

from pandaprobe.wrappers.anthropic import wrap_anthropic
from pandaprobe.wrappers.bedrock import wrap_bedrock
from pandaprobe.wrappers.gemini import wrap_gemini
from pandaprobe.wrappers.mistral import wrap_mistral
from pandaprobe.wrappers.openai import wrap_openai

__all__ = [
    "wrap_anthropic",
    "wrap_bedrock",
    "wrap_gemini",
    "wrap_mistral",
    "wrap_openai",
]
