PYTHON ?= python3.11
LAUNCH := $(PYTHON) experiments/scripts/launch.py
VALIDATE := $(PYTHON) experiments/scripts/validate_config.py

.PHONY: validate preflight run sweep status watch budget compare gc test

validate:
	$(VALIDATE) $(CONFIG)

preflight:
	$(LAUNCH) preflight $(HOST)

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
	pytest experiments/scripts/tests/test_infra.py
