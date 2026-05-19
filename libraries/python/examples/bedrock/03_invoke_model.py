"""AWS Bedrock wrapper — legacy InvokeModel API with automatic tracing.

Demonstrates wrap_bedrock instrumenting a Bedrock InvokeModel call against
the Anthropic Claude JSON body shape.  Prefer the Converse API for new
integrations — Converse is provider-agnostic and produces richer
instrumentation; this example exists to show the fallback path for
provider-specific bodies.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_REGION="us-east-1"

Run:
    uv run python examples/bedrock/03_invoke_model.py
"""

import json
import os

import boto3

import pandaprobe
from pandaprobe.wrappers import wrap_bedrock

client = wrap_bedrock(boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")))

if __name__ == "__main__":
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "system": "You are a concise assistant.",
            "messages": [
                {"role": "user", "content": "Explain what a Python decorator is in one sentence."},
            ],
        }
    )

    response = client.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    parsed = json.loads(response["body"].read())
    text = "".join(block.get("text", "") for block in parsed.get("content", []))
    usage = parsed.get("usage", {})

    print(f"Tokens: {usage.get('input_tokens')} input, {usage.get('output_tokens')} output")
    print(f"\nResponse:\n{text}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
