PYTHON ?= python

.PHONY: install test selftest estimate demo

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

selftest:
	$(PYTHON) -m shakedown selftest

estimate:
	$(PYTHON) -m shakedown estimate

demo: selftest
	$(PYTHON) -m shakedown demo --runs 12 --max-runs 12 --min-runs 8 --workers 3 --max-usd 15
