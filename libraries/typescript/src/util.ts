/** Return true if *value* is a thenable (Promise-like). */
export function isThenable(value: unknown): value is PromiseLike<unknown> {
  return (
    value !== null &&
    (typeof value === "object" || typeof value === "function") &&
    typeof (value as { then?: unknown }).then === "function"
  );
}

/**
 * Best-effort JSON-safe serialization of an arbitrary value.
 *
 * Primitives pass through; bigint → string; arrays recurse; objects prefer a
 * `toJSON()` hook (the Pydantic `model_dump()` analog) and otherwise fall back
 * to a key walk that skips `_`-prefixed (private/internal) keys — so framework
 * SDK objects don't leak internal state. Shared by the wrapper and integration
 * layers (re-exported from each `base.ts`) so the two never drift.
 */
export function safeSerialize(obj: unknown): unknown {
  if (obj === null || obj === undefined) {
    return obj;
  }
  const t = typeof obj;
  if (t === "string" || t === "number" || t === "boolean") {
    return obj;
  }
  if (t === "bigint") {
    return String(obj);
  }
  if (Array.isArray(obj)) {
    return obj.map((v) => safeSerialize(v));
  }
  if (t === "object") {
    const toJson = (obj as { toJSON?: () => unknown }).toJSON;
    if (typeof toJson === "function") {
      try {
        return safeSerialize(toJson.call(obj));
      } catch {
        // fall through to the key walk
      }
    }
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      if (!k.startsWith("_")) {
        out[k] = safeSerialize(v);
      }
    }
    return out;
  }
  return String(obj);
}
