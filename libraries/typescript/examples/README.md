# PandaProbe TypeScript SDK Examples

Real end-to-end examples that call live services and send traces to the PandaProbe backend. Use these to validate SDK integrations during development.

## Setup

### 1. Install and build the SDK

From `libraries/typescript`:

```bash
make ts-install   # or: pnpm install
pnpm run build    # examples import the built package via self-reference
```

### 2. Export environment variables

All examples require:

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_PROJECT_NAME="my-project"
export PANDAPROBE_ENDPOINT="http://localhost:8000"

# Provider keys (set whichever you need):
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."
export MISTRAL_API_KEY="..."
```

The SDK auto-initializes from these environment variables — no `init()` call is needed.

#### AWS Bedrock

The Bedrock example uses the AWS SDK for JavaScript credential chain. Set the region and configure exactly one of these authentication methods.

Bedrock API key:

```bash
export AWS_REGION="us-east-1"
export AWS_BEARER_TOKEN_BEDROCK="..."
```

AWS access keys:

```bash
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."  # only for temporary credentials
```

Shared AWS profile or IAM Identity Center:

```bash
export AWS_REGION="us-east-1"
export AWS_PROFILE="my-profile"
aws sso login --profile "$AWS_PROFILE"  # only for an SSO profile
aws sts get-caller-identity --profile "$AWS_PROFILE"  # verify credentials resolve
```

Claude Sonnet 5 uses the global inference profile by default:

```bash
export AWS_BEDROCK_MODEL_ID="global.anthropic.claude-sonnet-5"  # optional; this is the example default
```

The AWS identity needs `bedrock:InvokeModel` permission. Before the first Anthropic invocation, complete the Anthropic first-time-use form in the Bedrock model catalog and ensure the account can subscribe to the model. See the [AWS credential-chain documentation](https://docs.aws.amazon.com/sdk-for-javascript/v3/developer-guide/setting-credentials-node.html) and [Claude Sonnet 5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html).

Optional:

```bash
export PANDAPROBE_DEBUG=true     # debug logging
export PANDAPROBE_ENABLED=false  # disable tracing
```

### 3. Run any example with tsx

```bash
cd libraries/typescript
pnpm exec tsx examples/openai/chat.ts
```

Each example also installs the relevant provider/framework SDK as needed — install it
on demand, e.g. `pnpm add openai` or `pnpm add @langchain/langgraph @langchain/openai`.

## Layout

```
examples/
  decorators/          @trace / @span method decorators
  context-managers/    withTrace / withSpan callback wrappers
  openai/  anthropic/  gemini/  bedrock/  mistral/      LLM provider wrappers
  langchain/  langgraph/  deepagents/                   LangChain-family integrations
  claude-agent-sdk/  openai-agents/  vercel-ai/         agent-framework integrations
```

Always call `await flush()` (or `await shutdown()`) before the process exits so buffered
traces are sent.
