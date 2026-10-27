.PHONY: install install-reflect install-base install-langgraph install-langchain install-deepagents install-claude-agent-sdk install-openai-agents install-vercel-ai lint format format-check typecheck test test-cov build clean

# Base install: dev tooling and Reflect shell/lock test runtimes
install:
	pnpm install

# Explicit Reflect install, matching Python's install-reflect target.
install-reflect:
	pnpm add -D --save-exact @everruns/bashkit@0.19.0 fs-native-extensions@1.5.1

install-base:
	pnpm add -D openai @anthropic-ai/sdk @google/genai @mistralai/mistralai @aws-sdk/client-bedrock-runtime @langchain/core @langchain/openai

# Agent frameworks — install ONE at a time (the analog of Python's
# `uv sync --extra <name>`). Each adds the framework to devDependencies; install
# only the one you are working with rather than pulling every heavy framework.
# Reset with: git checkout -- package.json pnpm-lock.yaml && make install
install-langgraph:
	pnpm add -D @langchain/langgraph @langchain/openai @langchain/core

install-langchain:
	pnpm add -D langchain @langchain/openai @langchain/core

install-deepagents:
	pnpm add -D deepagents @langchain/core @langchain/openai \
		@langchain/langgraph @langchain/langgraph-checkpoint \
		@langchain/langgraph-sdk langchain "langsmith@>=0.7.1 <0.10.0"

install-claude-agent-sdk:
	pnpm add -D @anthropic-ai/claude-agent-sdk

install-openai-agents:
	pnpm add -D @openai/agents

install-vercel-ai:
	pnpm add -D ai @ai-sdk/openai

lint:
	pnpm exec biome check .

format:
	pnpm exec biome format --write .

format-check:
	pnpm exec biome format .

typecheck:
	pnpm exec tsc --noEmit

test:
	pnpm exec vitest run

test-cov:
	pnpm exec vitest run --coverage

build:
	pnpm run build

clean:
	rm -rf dist coverage

# Reflect runtimes are dev dependencies for real shell/lock tests; native frameworks
# remain in a separate pinned fixture so the base install supports Node 18.
.PHONY: install-native-test test-native test-package
install-native-test:
	npm ci --prefix test/native --ignore-scripts

test-native:
	pnpm test:native

test-package:
	pnpm run build
	pnpm test:package
