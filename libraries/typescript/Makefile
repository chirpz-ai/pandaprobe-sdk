.PHONY: install install-base install-langgraph install-langchain install-deepagents install-claude-agent-sdk install-openai-agents install-vercel-ai lint format format-check typecheck test test-cov build clean

# Base install: dev tooling
install:
	pnpm install

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
	pnpm add -D deepagents @langchain/core

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
