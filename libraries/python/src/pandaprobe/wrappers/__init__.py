"""LLM client wrappers for automatic span instrumentation.

Each provider lives in its own sub-package (e.g. ``wrappers/openai/``).
Adding a new provider (``wrap_anthropic``, ``wrap_google``, etc.) only
requires creating a new sub-package — no changes to core code.
"""

from pandaprobe.wrappers.openai import wrap_openai

__all__ = ["wrap_openai"]
