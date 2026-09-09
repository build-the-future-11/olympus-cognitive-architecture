.PHONY: test evidence external-analysis structural-readiness scientific-readiness readiness protocol v5-source integrity paper verify

PYTHON ?= ../../.venv/bin/python

test:
	PYTHONPATH="$(CURDIR)/src" $(PYTHON) -m pytest -q

evidence:
	$(PYTHON) scripts/validate_evidence.py

external-analysis:
	$(PYTHON) analysis/analyze_external_rigor.py

protocol:
	$(PYTHON) scripts/write_local_ood_protocol.py --manifest external/corebench/manifest.jsonl --verify

structural-readiness:
	$(PYTHON) scripts/publication_gate.py

scientific-readiness: external-analysis structural-readiness
	$(PYTHON) poststudy/scientific_readiness_gate.py --legacy-gate poststudy/STRUCTURAL_READINESS.json

readiness: scientific-readiness

v5-source:
	$(PYTHON) scripts/v5_source_manifest.py

integrity: readiness v5-source
	$(PYTHON) scripts/scoped_integrity.py

paper:
	mkdir -p output/pdf
	tectonic --keep-logs --outdir output/pdf paper/main.tex
	mv output/pdf/main.pdf output/pdf/pantheon-canonical-auditing.pdf

verify: test evidence readiness v5-source integrity paper
