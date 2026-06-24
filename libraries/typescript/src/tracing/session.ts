/**
 * AsyncLocalStorage-based session ID and user ID propagation.
 *
 * Mirrors the Python `tracing/session.py` ContextVar mechanism: a universal way
 * to set/retrieve the current session ID and user ID across all SDK layers
 * (decorators, wrappers, integrations) without explicit parameter passing.
 *
 * Two propagation styles:
 *  - **Scoped (recommended):** `runWithSession` / `runWithUser` use
 *    `AsyncLocalStorage.run()`, so each scope is isolated — concurrent async
 *    work (multiple conversations on one process) never overwrites or inherits
 *    the wrong id. This is the true analog of Python's `ContextVar` + the
 *    `with session(...)` context manager.
 *  - **Imperative:** `setCurrentSessionId` / `setCurrentUserId` use
 *    `enterWith()` to set an ambient value for the rest of the current async
 *    context. Convenient for top-of-request setup, but NOT safe to mutate from
 *    concurrent tasks sharing an async frame — prefer the scoped helpers there.
 */

import { AsyncLocalStorage } from "node:async_hooks";

interface SessionStore {
  sessionId: string | null;
  userId: string | null;
}

const storage = new AsyncLocalStorage<SessionStore>();

/** Opaque token returned by the imperative setters, used to restore the previous value. */
export type SessionToken = SessionStore | undefined;

function current(): SessionStore {
  return storage.getStore() ?? { sessionId: null, userId: null };
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

/** Return the session ID set in the current context, or `null`. */
export function getCurrentSessionId(): string | null {
  return current().sessionId;
}

/** Return the user ID set in the current context, or `null`. */
export function getCurrentUserId(): string | null {
  return current().userId;
}

// ---------------------------------------------------------------------------
// Scoped propagation (concurrency-safe) — the recommended form
// ---------------------------------------------------------------------------

/** Run *fn* with *sessionId* set for its scope, inheriting the current user ID. */
export function runWithSession<T>(sessionId: string | null, fn: () => T): T {
  return storage.run({ sessionId, userId: current().userId }, fn);
}

/** Run *fn* with *userId* set for its scope, inheriting the current session ID. */
export function runWithUser<T>(userId: string | null, fn: () => T): T {
  return storage.run({ sessionId: current().sessionId, userId }, fn);
}

// ---------------------------------------------------------------------------
// Imperative propagation (ambient; not concurrency-safe — see module docstring)
// ---------------------------------------------------------------------------

/** Set the session ID for the current async context. Returns a reset token. */
export function setCurrentSessionId(sessionId: string | null): SessionToken {
  const prev = storage.getStore();
  storage.enterWith({ sessionId, userId: prev?.userId ?? null });
  return prev;
}

/** Reset the session/user store to its previous value using the given token. */
export function resetCurrentSessionId(token: SessionToken): void {
  storage.enterWith(token ?? { sessionId: null, userId: null });
}

/** Set the user ID for the current async context. Returns a reset token. */
export function setCurrentUserId(userId: string | null): SessionToken {
  const prev = storage.getStore();
  storage.enterWith({ sessionId: prev?.sessionId ?? null, userId });
  return prev;
}

/** Reset the session/user store to its previous value using the given token. */
export function resetCurrentUserId(token: SessionToken): void {
  storage.enterWith(token ?? { sessionId: null, userId: null });
}
