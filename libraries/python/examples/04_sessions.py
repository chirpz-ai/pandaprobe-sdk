"""Example: Session-based tracing (grouping related traces).

Run with:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="sessions-example"
    python examples/04_sessions.py

Optionally override the backend URL:
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
"""

import time

import pandaprobe

client = pandaprobe.Client(debug=True)

if __name__ == "__main__":
    session = client.session("conversation-abc-123")

    # Turn 1
    with session.trace("turn-1", input={"query": "What is Python?"}) as t1:
        time.sleep(0.1)
        t1.set_output({"answer": "Python is a programming language."})

    # Turn 2
    with session.trace("turn-2", input={"query": "What about its type system?"}) as t2:
        time.sleep(0.1)
        t2.set_output({"answer": "Python uses dynamic typing with optional type hints."})

    # Turn 3
    with session.trace("turn-3", input={"query": "Thanks!"}) as t3:
        time.sleep(0.05)
        t3.set_output({"answer": "You're welcome!"})

    print(f"Session: {session.session_id}")
    print(f"Traces: {t1.trace_id}, {t2.trace_id}, {t3.trace_id}")
    client.flush()
    client.shutdown()
    print("All 3 traces sent under the same session.")
