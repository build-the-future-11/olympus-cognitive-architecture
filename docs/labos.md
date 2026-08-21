# LabOS Portfolio Layer

LabOS is the portfolio operating layer embedded in Olympus. It discovers sibling
repositories in the parent workspace, validates declared or inferred manifests,
tracks runs in a local SQLite database, emits structured event logs, and writes
portfolio-wide completion artifacts.

## Supported commands

```bash
.venv/bin/python -m olympus.cli labos discover --workspace ..
.venv/bin/python -m olympus.cli labos validate --workspace ..
.venv/bin/python -m olympus.cli labos run-smoke --workspace .. --project olympus
.venv/bin/python -m olympus.cli labos aggregate-metrics --workspace ..
.venv/bin/python -m olympus.cli labos generate-report --workspace .. --output-root .
```

## Manifest format

Each project may declare `labos.project.yaml` with:

- `project_id`, `title`, `description`, `hypothesis`
- `root_path`, `domain`, `language`, `manifest_source`
- `entry_points` with `name`, `command`, `profile`, and `expected_outputs`
- `datasets`, `expected_outputs`, `validation_commands`, `tags`
- `resource_requirements`

If a manifest is missing, LabOS infers one from Git presence, README headings,
`pyproject.toml`, `package.json`, and lightweight heuristics.

## Artifacts

LabOS writes:

- `artifacts/labos_runs.sqlite3`
- `artifacts/labos_events.jsonl`
- `artifacts/reports/portfolio_status.json`
- `artifacts/reports/PORTFOLIO_COMPLETION_REPORT.md`
- `artifacts/reports/REMAINING_EXTERNAL_ACTIONS.md`

LabOS reports contain local project paths and may contain captured command
output. The `artifacts/` directory is intentionally ignored by Git. Review and
redact generated reports before sharing them outside the machine where they
were produced.
