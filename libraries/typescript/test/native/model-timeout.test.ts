import { expect, it, vi } from "vitest";
import { nativeFactory } from "../../examples/reflect/_support/native.js";
import { Execution } from "../../src/reflect/index.js";
import { fixture } from "../reflect/helpers.js";

it.each([
  { framework: "langchain", outcome: "default timeout" },
  { framework: "langgraph", outcome: "default timeout" },
  { framework: "deepagents", outcome: "default timeout" },
  { framework: "langgraph", outcome: "slow response" },
  { framework: "langgraph", outcome: "extended timeout" },
  { framework: "langgraph", outcome: "caller cancellation" },
] as const)("$framework respects the example provider deadline: $outcome", async ({ framework, outcome }) => {
  vi.stubEnv("OPENAI_API_KEY", "offline-fixture");
  vi.stubEnv("OPENAI_BASE_URL", "http://openai.invalid/v1");
  vi.stubEnv("OPENAI_API_BASE", "http://openai.invalid/v1");
  vi.stubEnv("LANGSMITH_TRACING", "false");
  vi.stubEnv("LANGCHAIN_TRACING_V2", "false");
  try {
    const f = fixture();
    let factory = await nativeFactory(f.runtime, framework, {
      model: "gpt-6-luna",
      ...(outcome === "default timeout" ? {} : { modelTimeoutMs: 120_000 }),
    });
    // Fresh-runtime factories must retain the example's explicit deadline.
    if (outcome === "slow response") factory = await factory.forRuntime(f.runtime);
    const domain = vi.fn(() => "submitted");
    const agent = await factory(
      [{ name: "submit", description: "Submit a business action.", fields: {}, execute: domain }],
      "Answer the question.",
      { validation: true },
    );
    const controller = new AbortController();
    const execution = await Execution.create(f.runtime, {
      sessionId: "deadline",
      role: "validation",
      signal: controller.signal,
    });
    let started!: () => void;
    const requestStarted = new Promise<void>((resolve) => {
      started = resolve;
    });
    let respond!: () => void;
    let requestSignal!: AbortSignal;
    let attempts = 0;
    globalThis.fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const request = new Request(input, init);
      if (new URL(request.url).host === "fixture") return new Response(null, { status: 202 });
      expect(request.url).toBe("http://openai.invalid/v1/chat/completions");
      attempts++;
      requestSignal = request.signal;
      return new Promise<Response>((resolve, reject) => {
        const abort = () => reject(new DOMException("Aborted", "AbortError"));
        requestSignal.addEventListener("abort", abort, { once: true });
        respond = () => {
          requestSignal.removeEventListener("abort", abort);
          resolve(
            Response.json({
              id: "offline-completion",
              object: "chat.completion",
              created: 1,
              model: "gpt-6-luna",
              choices: [{ index: 0, message: { role: "assistant", content: "Answered." }, finish_reason: "stop" }],
              usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
            }),
          );
        };
        started();
      });
    }) as typeof fetch;
    // Only HTTP and the clock are fake. The real provider client must abort its
    // fetch; a response after 90 seconds must survive only the explicit override.
    vi.useFakeTimers();
    const result = execution
      .run(() => f.runtime.run(agent, "Question?", { sessionId: execution.sessionId }))
      .then(
        (value) => ({ value, error: undefined }),
        (error: Error) => ({ value: undefined, error }),
      );
    await requestStarted;
    if (outcome === "default timeout") {
      await vi.advanceTimersByTimeAsync(59_999);
      expect(requestSignal.aborted).toBe(false);
      await vi.advanceTimersByTimeAsync(1);
    } else if (outcome === "caller cancellation") {
      controller.abort(new Error("Caller stopped the replay"));
    } else {
      await vi.advanceTimersByTimeAsync(90_000);
      expect(requestSignal.aborted).toBe(false);
      if (outcome === "slow response") respond();
      else await vi.advanceTimersByTimeAsync(30_000);
    }
    const completed = await result;
    if (outcome === "slow response") {
      expect(completed.error).toBeUndefined();
      expect(completed.value.messages.at(-1).content).toBe("Answered.");
      expect(requestSignal.aborted).toBe(false);
    } else {
      expect(requestSignal.aborted).toBe(true);
      expect(completed.error).toMatchObject(
        outcome === "caller cancellation"
          ? { message: "Caller stopped the replay" }
          : { name: "TimeoutError", message: "Request timed out." },
      );
    }
    expect(attempts).toBe(1);
    expect(domain).not.toHaveBeenCalled();
    expect(f.runtime.workspace.loadExecution(execution.id)?.status).toBe(
      outcome === "slow response" ? "completed" : "failed",
    );
  } finally {
    vi.useRealTimers();
    vi.unstubAllEnvs();
  }
});
