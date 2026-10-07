# Contributing

Install the development extra, then run the same checks used by CI:

```bash
python -m pip install -e '.[dev]'
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
```

Method changes belong in `models/`, `tasks/` or `losses.py`. Runtime changes belong in
`engine/`. A new experiment usually needs only a YAML preset or sweep specification.
Use the smallest extension boundary that expresses the change; keep the trainer
independent of model families and research objectives.

Preserve model initialization and class ordering when comparing ablations. Report
random seeds, the real train/validation/test split, generated-pool provenance and
the configuration used to select the checkpoint. Add a behavioral test when a
change affects numerical semantics, checkpoint compatibility or run recovery.

Files under `reference/`, `models/recovered/` and `vendor/` are provenance-bearing
source snapshots. Implement adaptations outside them, or explicitly record the
change in `docs/source_inventory.json` and the reproduction notes.

See [architecture](docs/architecture.md) for task, callback and component examples.
