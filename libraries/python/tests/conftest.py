"""Test-suite-wide environment hygiene.

These fixtures protect tests from accidentally:

* sending data to a real PandaProbe backend, and
* picking up developer-machine values for ``PANDAPROBE_*`` env vars.

CI runs in a clean shell, but developers commonly have ``PANDAPROBE_API_KEY``,
``PANDAPROBE_PROJECT_NAME``, ``PANDAPROBE_ENDPOINT`` (sometimes pointed at the
production URL), and tuning vars like ``PANDAPROBE_FLUSH_INTERVAL`` exported.
Without isolation, those values silently leak into ``Client.__init__`` and can
either alter behavior (flush timing, batch size, debug logging) or — worse —
make a default-endpoint ``Client`` post real traces to production.

A single function-scoped autouse fixture handles all of this:

1. **Clears** every ``PANDAPROBE_*`` config var the SDK reads (config.py),
   so tests get a deterministic blank-slate environment regardless of host.
2. **Pins** ``PANDAPROBE_ENDPOINT`` to a fake host (``http://testserver``) for
   the duration of the test, so any default-endpoint ``Client`` created in the
   suite is incapable of reaching production. Tests that mock HTTP via
   ``respx`` already use this hostname; un-mocked accidents fail-fast at DNS.

Tests that need specific values opt in per-test via ``monkeypatch.setenv``,
which composes correctly with this fixture (function-scoped monkeypatch sets
take precedence; pytest tears them down before this fixture finalizes).
"""

from __future__ import annotations

import pytest

# Vars to clear (in order of definition in pandaprobe.config). These cover
# every env var read by ``resolve_config``.
_PANDAPROBE_CONFIG_VARS = (
    "PANDAPROBE_API_KEY",
    "PANDAPROBE_PROJECT_NAME",
    # PANDAPROBE_ENDPOINT is intentionally NOT in this list — it's pinned below.
    "PANDAPROBE_ENVIRONMENT",
    "PANDAPROBE_RELEASE",
    "PANDAPROBE_ENABLED",
    "PANDAPROBE_BATCH_SIZE",
    "PANDAPROBE_FLUSH_INTERVAL",
    "PANDAPROBE_MAX_QUEUE_SIZE",
    "PANDAPROBE_DEBUG",
)

_FAKE_ENDPOINT = "http://testserver"


@pytest.fixture(autouse=True)
def _isolate_pandaprobe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every ``PANDAPROBE_*`` config var and pin the endpoint to a fake host.

    Function-scoped + autouse: applies to every test in the suite. Uses
    ``monkeypatch`` so all changes are automatically reverted on test teardown.
    """
    for var in _PANDAPROBE_CONFIG_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PANDAPROBE_ENDPOINT", _FAKE_ENDPOINT)
