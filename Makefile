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
	$(PYTHON) -m shakedown demo --runs 60 --max-runs 40 --min-runs 16 --workers 3 --max-minutes 60 --max-usd 15
