# Changelog

## 0.2.0

- Replaced separate classification, diffusion and VQ training loops with one task-driven engine.
- Added sample-weighted gradient accumulation, autocast, clipping, callbacks and full runtime checkpoints.
- Introduced typed hierarchical YAML composition, defaults, interpolation and checked overrides.
- Extracted backbone feature contracts, fusion registries and recovered model compatibility adapters.
- Added artifact-dependent pipelines, content verification, stage recovery and experiment grids.
- Added multi-seed CSV/JSON/Markdown reports and reorganized data interfaces by responsibility.
- Added method, training and implementation documentation.
- Collected v1 flat configurations in `configs/legacy/` and class/path examples in `configs/data/`.

## 0.1.0

- Added classifier and generator source with file-level provenance.
- Added strict legacy checkpoint conversion, independent evaluation and a CPU method smoke test.
