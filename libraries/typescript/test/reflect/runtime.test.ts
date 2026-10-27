import { expect, it } from "vitest";
import { type Invocation, Outcome, ReplayCase } from "../../src/reflect/index.js";
import { captured } from "../helpers.js";
import { WRITE_RULE, fixture } from "./helpers.js";
it("returns the same task object and re-enters the same host with assigned evidence", async () => {
  const f = fixture();
  const result = { answer: "original" };
  const phases: string[] = [];
  const agent = async (request: Invocation) => {
    phases.push(request.phase);
    if (request.phase === "task") {
      f.server.addTrace(f.client);
      return result;
    }
    expect(request.previousResult).toBe(result);
    expect(await f.runtime.bash("cat /inbox/evidence.txt")).toContain("shipping");
    f.server.addTrace(f.client);
    await f.runtime.bash(WRITE_RULE);
    return "Changed guidance";
  };
  expect(await f.runtime.run(agent, "refund", { sessionId: "s", executionId: "task" })).toBe(result);
  expect(phases).toEqual(["task", "reflection"]);
  expect(f.runtime.report("task").reflection?.status).toBe("proposed");
  expect(f.runtime.workspace.rules()).toEqual({});
  await f.runtime.waitExport();
  const bodies = JSON.stringify(captured);
  expect(bodies).toContain('"role":"reflection"');
  expect(f.runtime.report("task").trace_ids).toHaveLength(1);
});
it.each(["throw", "timeout", "unchanged"])("reflection %s never replaces a task result", async (mode) => {
  const f = fixture({ reflectionTimeout: 0.01 });
  const agent = async (request: Invocation) => {
    if (request.phase === "task") {
      f.server.addTrace(f.client);
      return "answer";
    }
    if (mode === "throw") throw Error("checkpoint error");
    if (mode === "timeout") return new Promise<string>(() => {});
    return "no change";
  };
  expect(await f.runtime.run(agent, "x", { sessionId: "s", executionId: mode })).toBe("answer");
  expect(f.runtime.report(mode).reflection?.status).toBe(
    mode === "throw" ? "failed" : mode === "timeout" ? "timed_out" : "no_proposal",
  );
});
it("preserves original task exceptions, with evaluation but no automatic continuation", async () => {
  const f = fixture();
  const original = new Error("original task error");
  let calls = 0;
  await expect(
    f.runtime.run(
      async () => {
        calls++;
        f.server.addTrace(f.client);
        throw original;
      },
      "x",
      { sessionId: "s", executionId: "failure" },
    ),
  ).rejects.toBe(original);
  await f.runtime.settleEvaluation("failure");
  expect(calls).toBe(1);
  expect(f.runtime.report("failure").task_status).toBe("failed");
  expect(f.runtime.report("failure").reflection).toBeNull();
});
it("allows ordinary task writes and persists only compatible session memory", async () => {
  const f = fixture();
  await f.runtime.execution({ sessionId: "s", executionId: "memory" }, async () => {
    await f.runtime.bash("echo remembered >/context.md");
  });
  await f.runtime.execution({ sessionId: "s" }, async (e) => {
    expect(e.instructions()).toContain("remembered");
    await f.runtime.bash(`${WRITE_RULE}\necho unvalidated >/context.md`);
  });
  await f.runtime.execution({ sessionId: "s" }, async (e) => {
    expect(e.instructions()).not.toContain("unvalidated");
    expect(e.shell.rules()).toEqual({});
  });
  expect(f.runtime.proposals()).toHaveLength(1);
  expect(f.runtime.workspace.rules()).toEqual({});
});
it("isolates concurrent instances and sessions, including tool dispatch", async () => {
  const a = fixture();
  const b = fixture();
  const invoke = (f: typeof a, value: string) =>
    f.runtime.execution({ sessionId: value }, async (e) => {
      await f.runtime.bash(`echo ${value} >/context.md`);
      await new Promise((r) => setTimeout(r, 2));
      expect(e.shell.context()).toBe(`${value}\n`);
      expect(await f.runtime.bash("cat /context.md")).toContain(value);
    });
  await Promise.all([invoke(a, "one"), invoke(a, "two"), invoke(b, "three")]);
  expect(await a.runtime.bash("echo outside")).toContain("active");
});
it("separates caller deadlines from shared evaluation and cancels owned work at shutdown", async () => {
  const f = fixture({ evaluationTimeout: 1, shutdownTimeout: 0.05 });
  f.server.fault = "running";
  void f.runtime.submitTraces([f.server.addTrace()], { executionId: "pending", sessionId: "s" }).catch(() => {});
  expect((await f.runtime.settleEvaluation("pending", { timeout: 0.001 })).status).toBe("pending");
  expect(f.server.creates).toBe(1);
  await f.runtime.close();
  expect(f.runtime.report("pending").evaluation?.status).toBe("cancelled");
  expect((await f.api.evaluations.get([...f.server.runs.keys()][0]!)).id).toBeTruthy();
});
it("recovers late evaluation explicitly and persists a once-only reflection claim", async () => {
  const f = fixture();
  f.server.fault = "running";
  const result = await f.runtime.run(
    async (request: Invocation) => {
      if (request.phase === "task") f.server.addTrace(f.client);
      return "original";
    },
    "x",
    { sessionId: "s", executionId: "late" },
  );
  expect(result).toBe("original");
  expect(f.runtime.report("late").reflection).toBeNull();
  f.server.fault = "none";
  const next = fixture({}, f.server, f.workspace);
  await next.runtime.settleEvaluation("late");
  let calls = 0;
  const continuation = async () => {
    calls++;
    await next.runtime.bash(WRITE_RULE);
    return "proposal";
  };
  const outcomes = await Promise.all([
    next.runtime.continueReflection("late", continuation),
    next.runtime.continueReflection("late", continuation),
  ]);
  expect(outcomes[0].status).toBe("proposed");
  expect(calls).toBe(1);
  expect(
    (
      await f.runtime.continueReflection("late", async () => {
        throw Error("must not rerun");
      })
    ).status,
  ).toBe("proposed");
});
it("bounds shutdown for an uncooperative continuation and closes its late shell", async () => {
  const f = fixture({ reflectionTimeout: 0.01, shutdownTimeout: 0.01 });
  let release!: () => void;
  const gate = new Promise<void>((r) => {
    release = r;
  });
  let late!: Promise<string>;
  await f.runtime.run(
    async (request: Invocation) => {
      if (request.phase === "task") {
        f.server.addTrace(f.client);
        return "original";
      }
      late = (async () => {
        await gate;
        f.server.addTrace(f.client);
        return f.runtime.bash(WRITE_RULE);
      })();
      return late;
    },
    "x",
    { sessionId: "s", executionId: "slow" },
  );
  await f.runtime.close();
  release();
  expect(await late).toBe("");
  expect(f.runtime.proposals()).toEqual([]);
});
it("accepts explicit external trace references and handles memory-only reflection", async () => {
  const f = fixture();
  await f.runtime.submitTraces([f.server.addTrace()], { executionId: "external", sessionId: "s" });
  expect(
    (
      await f.runtime.continueReflection("external", async () => {
        await f.runtime.bash("echo reflection-memory >/context.md");
        return "no rule changes";
      })
    ).status,
  ).toBe("no_proposal");
  await f.runtime.execution({ sessionId: "s" }, async (e) => {
    expect(e.instructions()).toContain("reflection-memory");
  });
});
it("keeps completed scopes pinned until stream consumption ends", async () => {
  const f = fixture();
  await f.runtime.execution({ sessionId: "stream", executionId: "stream" }, async (e) => {
    async function* stream() {
      yield "first";
      await Promise.resolve();
      f.server.addTrace(f.client);
      yield "last";
    }
    for await (const _ of stream()) expect(e.taskStatus).toBe("running");
  });
  expect(f.runtime.report("stream").trace_ids).toHaveLength(1);
});

it("supports arbitrary prototype-like execution IDs without inherited dictionary entries", async () => {
  const f = fixture();
  for (const id of ["__proto__", "constructor", "toString"]) {
    await f.runtime.execution({ sessionId: id, executionId: id }, async (e) => {
      expect(e.id).toBe(id);
      return "done";
    });
    expect(f.runtime.report(id).task_status).toBe("completed");
  }
});
it("shutdown cancels a task even when it has a separate caller signal", async () => {
  const f = fixture({ shutdownTimeout: 0.02 });
  let entered!: () => void;
  const ready = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const promise = f.runtime.execution({ sessionId: "external", signal: new AbortController().signal }, async () => {
    entered();
    return new Promise(() => {});
  });
  const settled = expect(promise).rejects.toMatchObject({ name: "AbortError" });
  await ready;
  await f.runtime.close();
  await settled;
});

it("does not invoke a task with an already-aborted caller signal", async () => {
  const f = fixture();
  const controller = new AbortController();
  controller.abort();
  let calls = 0;
  await expect(
    f.runtime.execution({ sessionId: "cancelled", signal: controller.signal }, async () => {
      calls++;
      throw Error("must not run");
    }),
  ).rejects.toMatchObject({ name: "AbortError" });
  await new Promise((resolve) => setTimeout(resolve, 5));
  expect(calls).toBe(0);
});
