"""AWS Bedrock wrapper — Converse streaming with automatic tracing.

Demonstrates wrap_bedrock handling a streamed Converse response.  The
wrapper collects chunks (``contentBlockDelta`` events) and reduces them
into a single LLM span with the full output and final token usage.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_REGION="us-east-1"

Run:
    uv run python examples/bedrock/02_converse_stream.py
"""

import os

import boto3

import pandaprobe
from pandaprobe.wrappers import wrap_bedrock

client = wrap_bedrock(boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")))

if __name__ == "__main__":
    print("Streaming response:\n")

    response = client.converse_stream(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        system=[{"text": "You are a helpful assistant."}],
        messages=[
            {"role": "user", "content": [{"text": "Write a short poem about debugging code."}]},
        ],
        inferenceConfig={"temperature": 0.7, "maxTokens": 200},
    )

    for event in response["stream"]:
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        text = delta.get("text")
        if text:
            print(text, end="", flush=True)

    print("\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("Trace sent to PandaProbe backend.")
