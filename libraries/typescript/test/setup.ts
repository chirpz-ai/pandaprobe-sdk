/**
 * Global test setup: isolate PANDAPROBE_* env vars, pin the endpoint to a fake
 * host, reset the client singleton between tests, and install the fetch mock.
 * Mirrors the Python `conftest.py` autouse fixture.
 */

import { afterEach, beforeEach } from "vitest";
import { _resetClientForTests } from "../src/client.js";
import { setCurrentSessionId, setCurrentUserId } from "../src/tracing/session.js";
import { installFetchMock } from "./helpers.js";

const PANDAPROBE_VARS = [
  "PANDAPROBE_API_KEY",
  "PANDAPROBE_PROJECT_NAME",
  "PANDAPROBE_ENDPOINT",
  "PANDAPROBE_ENVIRONMENT",
  "PANDAPROBE_RELEASE",
  "PANDAPROBE_ENABLED",
  "PANDAPROBE_BATCH_SIZE",
  "PANDAPROBE_FLUSH_INTERVAL",
  "PANDAPROBE_MAX_QUEUE_SIZE",
  "PANDAPROBE_DEBUG",
];

beforeEach(() => {
  for (const v of PANDAPROBE_VARS) {
    delete process.env[v];
  }
  process.env.PANDAPROBE_ENDPOINT = "http://testserver";
  _resetClientForTests();
  installFetchMock();
});

afterEach(() => {
  _resetClientForTests();
  // Clear any session/user contextvar leakage between tests.
  setCurrentSessionId(null);
  setCurrentUserId(null);
});
