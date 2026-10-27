PYTHON ?= python

.PHONY: install test selftest estimate baseline demo demo-quick report lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

selftest:
	$(PYTHON) -m shakedown selftest

estimate:
	$(PYTHON) -m shakedown estimate

demo: selftest
	$(PYTHON) -m shakedown demo --runs 60 --max-runs 60 --min-runs 30 --fix-runs 60 --workers 3 --max-minutes 60 --max-usd 15

baseline:
	$(PYTHON) -m shakedown run --baseline -n 60

demo-quick:
	$(PYTHON) -m shakedown demo --runs 30 --max-runs 20 --min-runs 12 --workers 3 --max-minutes 30 --max-usd 15 --no-fix

report:
	$(PYTHON) -m shakedown report $(RUN)

lint:
	ruff check src tests scripts

clean:
	$(PYTHON) -c "import shutil; from pathlib import Path; [shutil.rmtree(p, ignore_errors=True) for p in map(Path, ('build', 'dist', '.pytest_cache', '.ruff_cache', '.mypy_cache', 'shakedown.egg-info', 'shakedown_agent_profiler.egg-info'))]"
