import { Rule } from "./models.js";
/** Python-compatible sorted JSON used by revision and proposal hashes. */
export function json(value: unknown, options: { ascii?: boolean; spaced?: boolean; sortKeys?: boolean } = {}): string {
  const ancestors = new Set<object>();
  const quote = (s: string) => {
    const encoded = JSON.stringify(s);
    return options.ascii
      ? encoded.replace(/[\u007f-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`)
      : encoded;
  };
  const comma = options.spaced ? ", " : ",";
  const colon = options.spaced ? ": " : ":";
  function encode(v: unknown): string {
    if (v === null) return "null";
    if (typeof v === "string") return quote(v);
    if (typeof v === "boolean") return String(v);
    if (typeof v === "number") {
      if (!Number.isFinite(v)) throw new TypeError("JSON evidence must be finite");
      return String(v);
    }
    if (typeof v !== "object" || v === undefined) throw new TypeError("Evidence must be reproducible JSON");
    if (ancestors.has(v)) throw new TypeError("Cyclic JSON evidence");
    ancestors.add(v);
    try {
      if (Array.isArray(v)) return `[${v.map(encode).join(comma)}]`;
      const prototype = Object.getPrototypeOf(v);
      if (prototype !== Object.prototype && prototype !== null && !(v instanceof Rule))
        throw new TypeError("Evidence must use plain JSON objects");
      const keys = Object.keys(v);
      if (options.sortKeys !== false)
        keys.sort((a, b) => {
          const x = Array.from(a);
          const y = Array.from(b);
          for (let i = 0; i < Math.min(x.length, y.length); i++) {
            const d = x[i]!.codePointAt(0)! - y[i]!.codePointAt(0)!;
            if (d) return d;
          }
          return x.length - y.length;
        });
      return `{${keys.map((k) => `${quote(k)}${colon}${encode((v as Record<string, unknown>)[k])}`).join(comma)}}`;
    } finally {
      ancestors.delete(v);
    }
  }
  return encode(value);
}
export const pythonJson = (value: unknown): string => json(value, { ascii: true, spaced: true });
