.PHONY: py-install py-install-dev py-sync py-lock py-lint py-format py-format-check py-test py-test-cov py-build py-clean py-nuke-venv

PYTHON_DIR = libraries/python

py-install:
	$(MAKE) -C $(PYTHON_DIR) install

py-install-dev:
	$(MAKE) -C $(PYTHON_DIR) install-dev

py-sync:
	$(MAKE) -C $(PYTHON_DIR) sync

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

py-nuke-venv:
	$(MAKE) -C $(PYTHON_DIR) nuke-venv
