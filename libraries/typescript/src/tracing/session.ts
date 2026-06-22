/**
 * AsyncLocalStorage-based session ID and user ID propagation.
 *
 * Mirrors the Python `tracing/session.py` ContextVar mechanism: a universal way
 * to set/retrieve the current session ID and user ID across all SDK layers
 * (decorators, wrappers, integrations) without explicit parameter passing.
 */

import { AsyncLocalStorage } from "node:async_hooks";

interface SessionStore {
  sessionId: string | null;
  userId: string | null;
}

const storage = new AsyncLocalStorage<SessionStore>();

/** Opaque token returned by setters, used to restore the previous value. */
export type SessionToken = SessionStore | undefined;

function current(): SessionStore {
  return storage.getStore() ?? { sessionId: null, userId: null };
}

// ---------------------------------------------------------------------------
// Session ID
// ---------------------------------------------------------------------------

/** Return the session ID set in the current context, or `null`. */
export function getCurrentSessionId(): string | null {
  return current().sessionId;
}

/** Set the session ID for the current context. Returns a reset token. */
export function setCurrentSessionId(sessionId: string | null): SessionToken {
  const prev = storage.getStore();
  storage.enterWith({ sessionId, userId: prev?.userId ?? null });
  return prev;
}

/** Reset the session/user store to its previous value using the given token. */
export function resetCurrentSessionId(token: SessionToken): void {
  storage.enterWith(token ?? { sessionId: null, userId: null });
}

// ---------------------------------------------------------------------------
// User ID
// ---------------------------------------------------------------------------

/** Return the user ID set in the current context, or `null`. */
export function getCurrentUserId(): string | null {
  return current().userId;
}

/** Set the user ID for the current context. Returns a reset token. */
export function setCurrentUserId(userId: string | null): SessionToken {
  const prev = storage.getStore();
  storage.enterWith({ sessionId: prev?.sessionId ?? null, userId });
  return prev;
}

/** Reset the session/user store to its previous value using the given token. */
export function resetCurrentUserId(token: SessionToken): void {
  storage.enterWith(token ?? { sessionId: null, userId: null });
}
