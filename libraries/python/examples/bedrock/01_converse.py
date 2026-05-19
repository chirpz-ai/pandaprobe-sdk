"""AWS Bedrock wrapper — Converse API blocking call with automatic tracing.

Demonstrates wrap_bedrock instrumenting a Bedrock Converse call.  The
wrapper automatically creates an LLM span capturing model, tokens, I/O, and
inference parameters.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_REGION="us-east-1"

Run:
    uv run python examples/bedrock/01_converse.py
"""

import os

import boto3

import pandaprobe
from pandaprobe.wrappers import wrap_bedrock

client = wrap_bedrock(boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")))

if __name__ == "__main__":
    response = client.converse(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        system=[{"text": "You are a concise assistant. Answer in one or two sentences."}],
        messages=[
            {"role": "user", "content": [{"text": "Explain what a Python decorator is."}]},
        ],
        inferenceConfig={"temperature": 0.5, "maxTokens": 200},
    )

    message = response["output"]["message"]
    text = "".join(block.get("text", "") for block in message["content"])
    usage = response.get("usage", {})

    print(f"Tokens: {usage.get('inputTokens')} input, {usage.get('outputTokens')} output")
    print(f"\nResponse:\n{text}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
