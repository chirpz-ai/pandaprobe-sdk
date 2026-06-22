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
});
