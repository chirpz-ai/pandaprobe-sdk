import { beforeEach, describe, expect, it } from "vitest";
import { flush, init, session, setSession, setUser, user, withTrace } from "../src/index.js";
import { requestsTo } from "./helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  const reqs = requestsTo("/traces");
  return reqs[reqs.length - 1]?.body as Record<string, unknown>;
}

describe("session / user propagation", () => {
  it("scoped session() applies to traces inside the scope", async () => {
    await session("conv-123", async () => {
      await withTrace("t", async () => {});
    });
    await flush();
    expect(lastTrace().session_id).toBe("conv-123");
  });

  it("scoped user() applies to traces inside the scope", async () => {
    await user("user-abc", async () => {
      await withTrace("t", async () => {});
    });
    await flush();
    expect(lastTrace().user_id).toBe("user-abc");
  });

  it("explicit trace options win over the contextvar", async () => {
    await session("ctx-session", async () => {
      await withTrace("t", { sessionId: "explicit" }, async () => {});
    });
    await flush();
    expect(lastTrace().session_id).toBe("explicit");
  });

  it("imperative setSession/setUser propagate", async () => {
    setSession("s1");
    setUser("u1");
    await withTrace("t", async () => {});
    await flush();
    const body = lastTrace();
    expect(body.session_id).toBe("s1");
    expect(body.user_id).toBe("u1");
  });

  it("nests session within user, inheriting both", async () => {
    await user("u-1", async () => {
      await session("s-1", async () => {
        await withTrace("t", async () => {});
      });
    });
    await flush();
    const body = lastTrace();
    expect(body.session_id).toBe("s-1");
    expect(body.user_id).toBe("u-1");
  });

  it("keeps concurrent session() scopes isolated (no cross-conversation leak)", async () => {
    // Regression: scoped session must use AsyncLocalStorage.run(), not enterWith,
    // so interleaved async work on one process never inherits the wrong id.
    await Promise.all([
      session("conv-A", async () => {
        await new Promise((r) => setTimeout(r, 15));
        await withTrace("trace-A", async () => {});
      }),
      session("conv-B", async () => {
        await new Promise((r) => setTimeout(r, 5));
        await withTrace("trace-B", async () => {});
      }),
    ]);
    await flush();

    const traces = requestsTo("/traces").map((r) => r.body as Record<string, unknown>);
    const a = traces.find((t) => t.name === "trace-A")!;
    const b = traces.find((t) => t.name === "trace-B")!;
    expect(a.session_id).toBe("conv-A");
    expect(b.session_id).toBe("conv-B");
  });
});
