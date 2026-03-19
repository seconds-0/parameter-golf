PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
LAUNCH := $(PYTHON) experiments/scripts/launch.py
VALIDATE := $(PYTHON) experiments/scripts/validate_config.py

.PHONY: validate preflight run sweep status watch budget compare gc test

validate:
	$(VALIDATE) $(CONFIG)

preflight:
	$(LAUNCH) preflight $(HOST) $(if $(CONFIG),--config $(CONFIG),)

run:
	$(LAUNCH) run $(CONFIG) --host $(HOST)

sweep:
	$(LAUNCH) sweep $(CONFIG) --hosts $(HOSTS)

status:
	$(LAUNCH) status

watch:
	$(LAUNCH) status --watch

budget:
	$(LAUNCH) budget

compare:
	$(PYTHON) experiments/scripts/compare.py experiments/results/*/metrics.json

gc:
	$(LAUNCH) gc $(HOST)

test:
	$(PYTHON) -m pytest experiments/scripts/tests/test_infra.py
