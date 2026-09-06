.PHONY: test evidence external-analysis protocol paper verify

PYTHON ?= ../../.venv/bin/python

test:
	PYTHONPATH="$(CURDIR)/src" $(PYTHON) -m pytest -q

evidence:
	$(PYTHON) scripts/validate_evidence.py

external-analysis:
	$(PYTHON) analysis/analyze_external_rigor.py

protocol:
	$(PYTHON) scripts/write_local_ood_protocol.py --manifest external/corebench/manifest.jsonl --verify

paper:
	mkdir -p output/pdf
	tectonic --keep-logs --outdir output/pdf paper/main.tex
	mv output/pdf/main.pdf output/pdf/pantheon-canonical-auditing.pdf

verify: test evidence external-analysis protocol paper
