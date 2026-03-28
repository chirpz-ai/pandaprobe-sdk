.PHONY: py-install py-install-crewai py-lock py-lint py-format py-format-check py-test py-test-cov py-build py-clean

PYTHON_DIR = libraries/python

py-install:
	$(MAKE) -C $(PYTHON_DIR) install

py-install-crewai:
	$(MAKE) -C $(PYTHON_DIR) install-crewai

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