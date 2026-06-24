.PHONY: py-install py-install-langgraph py-install-langchain py-install-deepagents py-install-google-adk py-install-claude-agent-sdk py-install-crewai py-install-openai-agents py-lock py-lint py-format py-format-check py-test py-test-cov py-build py-clean ts-install ts-install-base ts-install-langgraph ts-install-langchain ts-install-deepagents ts-install-claude-agent-sdk ts-install-openai-agents ts-install-vercel-ai ts-lint ts-format ts-format-check ts-typecheck ts-test ts-test-cov ts-build ts-clean

PYTHON_DIR = libraries/python
TYPESCRIPT_DIR = libraries/typescript

py-install:
	$(MAKE) -C $(PYTHON_DIR) install

py-install-langgraph:
	$(MAKE) -C $(PYTHON_DIR) install-langgraph

py-install-langchain:
	$(MAKE) -C $(PYTHON_DIR) install-langchain

py-install-deepagents:
	$(MAKE) -C $(PYTHON_DIR) install-deepagents

py-install-google-adk:
	$(MAKE) -C $(PYTHON_DIR) install-google-adk

py-install-claude-agent-sdk:
	$(MAKE) -C $(PYTHON_DIR) install-claude-agent-sdk

py-install-crewai:
	$(MAKE) -C $(PYTHON_DIR) install-crewai

py-install-openai-agents:
	$(MAKE) -C $(PYTHON_DIR) install-openai-agents

py-lock:
	$(MAKE) -C $(PYTHON_DIR) lock

py-lint:
	$(MAKE) -C $(PYTHON_DIR) lint

py-format:
	$(MAKE) -C $(PYTHON_DIR) format

py-format-check:
	$(MAKE) -C $(PYTHON_DIR) format-check

py-test:
	$(MAKE) -C $(PYTHON_DIR) test

py-test-cov:
	$(MAKE) -C $(PYTHON_DIR) test-cov

py-build:
	$(MAKE) -C $(PYTHON_DIR) build

py-clean:
	$(MAKE) -C $(PYTHON_DIR) clean

# ---------------------------------------------------------------------------
# TypeScript SDK (libraries/typescript)
# ---------------------------------------------------------------------------

ts-install:
	$(MAKE) -C $(TYPESCRIPT_DIR) install

ts-install-base:
	$(MAKE) -C $(TYPESCRIPT_DIR) install-base

ts-install-langgraph:
	$(MAKE) -C $(TYPESCRIPT_DIR) install-langgraph

ts-install-langchain:
	$(MAKE) -C $(TYPESCRIPT_DIR) install-langchain

ts-install-deepagents:
	$(MAKE) -C $(TYPESCRIPT_DIR) install-deepagents

ts-install-claude-agent-sdk:
	$(MAKE) -C $(TYPESCRIPT_DIR) install-claude-agent-sdk

ts-install-openai-agents:
	$(MAKE) -C $(TYPESCRIPT_DIR) install-openai-agents

ts-install-vercel-ai:
	$(MAKE) -C $(TYPESCRIPT_DIR) install-vercel-ai

ts-lint:
	$(MAKE) -C $(TYPESCRIPT_DIR) lint

ts-format:
	$(MAKE) -C $(TYPESCRIPT_DIR) format

ts-format-check:
	$(MAKE) -C $(TYPESCRIPT_DIR) format-check

ts-typecheck:
	$(MAKE) -C $(TYPESCRIPT_DIR) typecheck

ts-test:
	$(MAKE) -C $(TYPESCRIPT_DIR) test

ts-test-cov:
	$(MAKE) -C $(TYPESCRIPT_DIR) test-cov

ts-build:
	$(MAKE) -C $(TYPESCRIPT_DIR) build

ts-clean:
	$(MAKE) -C $(TYPESCRIPT_DIR) clean