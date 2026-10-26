"""Run the complete Reflect loop locally, with no credentials or model calls.

    pip install -e '.[reflect]'
    python examples/reflect/offline_demo.py

A small HTTP fixture implements the platform contracts, including delayed trace
persistence and asynchronous evaluations. A deliberately faulty scripted agent
edits its rules through the real supplied Bash. An independent arithmetic
verifier tests fresh controls, candidates, and a protected successful case.
This demonstrates orchestration, not a claim about a real model's improvement.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from pandaprobe import Client
from pandaprobe.reflect import Outcome, Reflect, ReflectConfig, ReplayCase

RULE = """# Complete sums
Applies to: summing lists of numbers
Use every number in the input; compute sum(values) instead of summing only the first two.
"""


@contextmanager
def local_platform():
    """A test service, not a replacement backend or SDK runtime dependency."""
    state = {"traces": {}, "runs": {}, "reads": {}, "polls": {}, "creates": 0, "requests": 0}
    lock = threading.RLock()
    project_id = str(uuid4())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            state["requests"] += 1
            if self.headers.get("X-API-Key") != "offline-key" or self.headers.get("X-Project-Name") != "offline-demo":
                self.reply(401, {"detail": "Incorrect explicit SDK identity"})
                return False
            return True

        def do_POST(self):
            with lock:
                if not self.authorized():
                    return
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/traces":
                    data["spans"] = [{"trace_id": data["trace_id"], **span} for span in data.get("spans", [])]
                    state["traces"][data["trace_id"]] = {"project_id": project_id, **data}
                    self.reply(202, {"trace_id": data["trace_id"]})
                elif self.path == "/evaluations/trace-runs/batch":
                    run_id = str(uuid4())
                    state["creates"] += 1
                    state["runs"][run_id] = data
                    self.reply(202, run(run_id, "PENDING"))
                else:
                    self.reply(404, {"detail": "Unknown route"})

        def do_GET(self):
            with lock:
                if not self.authorized():
                    return
                if self.path.startswith("/traces/"):
                    trace_id = self.path.rsplit("/", 1)[1]
                    count = state["reads"].get(trace_id, 0)
                    state["reads"][trace_id] = count + 1
                    if count == 0 or trace_id not in state["traces"]:
                        self.reply(404, {"detail": "Accepted, but not persisted yet"})
                    else:
                        self.reply(200, state["traces"][trace_id])
                elif self.path.startswith("/evaluations/trace-runs/"):
                    parts = self.path.split("/")
                    run_id = parts[3]
                    if run_id not in state["runs"]:
                        self.reply(404, {})
                    elif self.path.endswith("/scores"):
                        self.reply(200, scores(run_id))
                    else:
                        state["polls"][run_id] = state["polls"].get(run_id, 0) + 1
                        self.reply(200, run(run_id, "COMPLETED" if state["polls"][run_id] >= 2 else "RUNNING"))
                else:
                    self.reply(404, {})

    def run(run_id, status):
        request = state["runs"][run_id]
        return {
            "id": run_id,
            "status": status,
            "metric_names": request["metrics"],
            "target_type": "TRACE",
            "total_targets": len(request["trace_ids"]),
            "evaluated_count": len(request["trace_ids"]) if status == "COMPLETED" else 0,
            "failed_count": 0,
        }

    def scores(run_id):
        request = state["runs"][run_id]
        result = []
        for trace_id in request["trace_ids"]:
            trace = state["traces"][trace_id]
            values = trace["metadata"]["values"]
            actual = int(trace["output"]["messages"][0]["content"])
            for metric in request["metrics"]:
                value = float(actual == sum(values)) if metric == "task_completion" else 1.0
                result.append(
                    {
                        "id": str(uuid4()),
                        "trace_id": trace_id,
                        "eval_run_id": run_id,
                        "name": metric,
                        "value": str(value),
                        "data_type": "NUMERIC",
                        "status": "SUCCESS",
                        "reason": "Compared the returned number with independently computed arithmetic",
                    }
                )
        return result

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


async def demonstrate(workspace: Path) -> dict:
    with local_platform() as (endpoint, service):
        sdk = Client(
            api_key="offline-key",
            project_name="offline-demo",
            endpoint=endpoint,
            enabled=True,
            batch_size=1,
            flush_interval=0.01,
        )
        try:
            async with Reflect(
                sdk,
                config=ReflectConfig(
                    workspace=workspace,
                    poll_interval=0.01,
                    cooldown_seconds=0,
                    trace_visibility_timeout=3,
                    evaluation_timeout=3,
                ),
            ) as reflect:

                async def agent(request):
                    if request.phase == "reflection":
                        with sdk.trace("scripted reflection") as trace:
                            await reflect.bash("cat /inbox/notice.json")
                            await reflect.bash("grep -rl 'Use every number' /rules || true")
                            await reflect.bash(
                                "cat > /rules/sums.md <<'EOF'\n" + RULE + "EOF\ncat /rules/sums.md > /context.md"
                            )
                            trace.set_output(
                                {"messages": [{"role": "assistant", "content": "Proposed complete summation"}]}
                            )
                            return "Proposed complete summation; independently verify it"
                    values = request.input
                    with sdk.trace(
                        "scripted sum",
                        input={"messages": [{"role": "user", "content": f"Sum {values}"}]},
                        metadata={"values": values},
                    ) as trace:
                        await reflect.bash("grep -rl 'summing lists' /rules || true")
                        guidance = await reflect.bash("cat /rules/*.md 2>/dev/null || true")
                        with trace.span("scripted decision", kind="LLM", model="offline-script") as span:
                            span.set_input(
                                {
                                    "messages": [
                                        {"role": "system", "content": request.instructions},
                                        {"role": "tool", "content": guidance},
                                    ]
                                }
                            )
                            # The scripted decision consumes this exact text.
                            result = sum(values) if "Use every number" in guidance else sum(values[:2])
                            span.set_output({"messages": [{"role": "assistant", "content": str(result)}]})
                        trace.set_output({"messages": [{"role": "assistant", "content": str(result)}]})
                        return result

                first = await reflect.run(agent, [2, 3, 4], session_id="customer", execution_id="failure")
                reflection = await reflect.settle_reflection("failure")
                if reflection.proposal is None:
                    raise AssertionError(
                        f"Expected a proposal, received {reflection}; {await reflect.settle_evaluation('failure')}"
                    )
                before_validation = list(reflect.workspace.rules())
                cases = [
                    ReplayCase(id="failure-one", input=[2, 3, 4]),
                    ReplayCase(id="failure-two", input=[10, 20, 30]),
                    ReplayCase(id="protected", input=[1, 1], protected=True),
                ]

                async def verify(request):
                    result = await request.run(agent)
                    expected = sum(request.case.input)
                    return Outcome(
                        scores={"task_completion": float(result == expected), "coherence": 1},
                        success=result == expected,
                        evidence={"actual": result, "expected": expected},
                    )

                verdict = await reflect.validate(reflection.proposal.id, cases=cases, replay=verify)
                later = await reflect.run(agent, [2, 3, 4], session_id="fresh-customer", execution_id="after")
                await reflect.wait_export(timeout=3)
                roles = [t["metadata"]["pandaprobe.reflect"]["role"] for t in service["traces"].values()]
                evaluated_roles = [
                    service["traces"][t]["metadata"]["pandaprobe.reflect"]["role"]
                    for run_request in service["runs"].values()
                    for t in run_request["trace_ids"]
                ]
                return {
                    "original_result": first,
                    "candidate_status": reflection.status,
                    "accepted_before_validation": before_validation,
                    "validation": verdict.outcome,
                    "validation_reason": verdict.reason,
                    "later_result": later,
                    "accepted_files": list(reflect.workspace.rules()),
                    "trace_roles": sorted(set(roles)),
                    "evaluated_roles": sorted(set(evaluated_roles)),
                    "evaluation_creates": service["creates"],
                    "poll_counts": list(service["polls"].values()),
                    "delayed_trace_reads": min(service["reads"].values()),
                    "workspace": str(reflect.workspace.path),
                    "real_model_calls": 0,
                }
        finally:
            await asyncio.to_thread(sdk.shutdown)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pandaprobe-reflect-demo-") as folder:
        print(json.dumps(asyncio.run(demonstrate(Path(folder))), indent=2))
