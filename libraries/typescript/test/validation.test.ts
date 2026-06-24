import { describe, expect, it, vi } from "vitest";
import {
  extractLastAssistantMessage,
  extractLastUserMessage,
  validateMessagesFormat,
  warnIfInvalidMessages,
} from "../src/validation.js";

describe("validateMessagesFormat", () => {
  it("accepts null/undefined silently", () => {
    expect(() => validateMessagesFormat(null, "x")).not.toThrow();
    expect(() => validateMessagesFormat(undefined, "x")).not.toThrow();
  });

  it("accepts a valid messages object", () => {
    expect(() => validateMessagesFormat({ messages: [{ role: "user", content: "hi" }] }, "x")).not.toThrow();
  });

  it("accepts array and null content", () => {
    expect(() =>
      validateMessagesFormat({ messages: [{ role: "user", content: [{ type: "text" }] }] }, "x"),
    ).not.toThrow();
    expect(() => validateMessagesFormat({ messages: [{ role: "tool", content: null }] }, "x")).not.toThrow();
  });

  it("rejects non-object, missing messages, bad shapes", () => {
    expect(() => validateMessagesFormat("nope", "x")).toThrow(/must be an object/);
    expect(() => validateMessagesFormat({ foo: 1 }, "x")).toThrow(/must contain a 'messages' key/);
    expect(() => validateMessagesFormat({ messages: "no" }, "x")).toThrow(/must be an array/);
    expect(() => validateMessagesFormat({ messages: [42] }, "x")).toThrow(/must be an object/);
    expect(() => validateMessagesFormat({ messages: [{ content: "x" }] }, "x")).toThrow(/missing required key 'role'/);
    expect(() => validateMessagesFormat({ messages: [{ role: "user" }] }, "x")).toThrow(
      /missing required key 'content'/,
    );
    expect(() => validateMessagesFormat({ messages: [{ role: 1, content: "x" }] }, "x")).toThrow(
      /'role'\] must be a string/,
    );
    expect(() => validateMessagesFormat({ messages: [{ role: "u", content: 5 }] }, "x")).toThrow(
      /'content'\] must be a string/,
    );
  });
});

describe("warnIfInvalidMessages", () => {
  it("never throws but warns on invalid input", () => {
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(() => warnIfInvalidMessages({ bad: true }, "trace input")).not.toThrow();
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});

describe("extractLastUserMessage", () => {
  it("returns only the last user/human message", () => {
    const input = {
      messages: [
        { role: "user", content: "first" },
        { role: "assistant", content: "reply" },
        { role: "user", content: "second" },
      ],
    };
    expect(extractLastUserMessage(input)).toEqual({ messages: [{ role: "user", content: "second" }] });
  });

  it("returns input unchanged when no match or wrong shape", () => {
    expect(extractLastUserMessage({ messages: [{ role: "assistant", content: "x" }] })).toEqual({
      messages: [{ role: "assistant", content: "x" }],
    });
    expect(extractLastUserMessage("raw")).toBe("raw");
  });
});

describe("extractLastAssistantMessage", () => {
  it("returns only the last assistant/ai message", () => {
    const out = {
      messages: [
        { role: "user", content: "q" },
        { role: "assistant", content: "a1" },
        { role: "ai", content: "a2" },
      ],
    };
    expect(extractLastAssistantMessage(out)).toEqual({ messages: [{ role: "ai", content: "a2" }] });
  });
});
