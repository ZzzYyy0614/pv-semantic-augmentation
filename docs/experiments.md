# Experiment composition and artifacts

## Pipeline stages

The paper pipeline declares five stages in `configs/pipelines/paper.yaml`.
Independent estimator and generator stages run in deterministic serial order.
Execution is currently single-process; dependency independence does not imply
concurrent GPU scheduling.

```yaml
- id: recognizer
  kind: classification
  config: ../tasks/recognizer.yaml
  inputs:
    teacher: '@estimator.checkpoint'
    generated_manifest: '@synthetic.manifest'
```

`@stage.artifact` both resolves an output path and introduces a dependency. Explicit
`depends_on` can add ordering constraints. Stage IDs cannot contain path separators.
The scheduler rejects unknown references and dependency cycles before execution.

Training stages accept config paths, a list of checked `overrides` and named
input paths. Their output directories are assigned by the pipeline. Sampling and
evaluation accept their required named inputs and a validated set of options.
Configurations and inheritance paths are file-relative; external dataset/weight
paths are workspace-relative. The dry plan resolves output references without
reading data or allocating models.

```bash
pvaug pipeline --config configs/pipelines/paper.yaml --dry-run
pvaug pipeline --config configs/pipelines/paper.yaml --output runs/paper-seed42
pvaug pipeline --config configs/pipelines/paper.yaml --output runs/paper-seed42 --resume
```

The pipeline writes:

```text
runs/paper/
├── plan.json
├── pipeline.json
├── estimator/     # config, environment, history, audit, best.pt, last.pt
├── generator/     # config, environment, history, audit, EMA checkpoint
├── synthetic/     # images, generated.csv, generation_metadata.json
├── recognizer/    # supervised/semantic history and checkpoints
└── evaluation/    # metrics.json, predictions.csv, audit.json
```

State transitions are `running → completed` or `running → failed`. Atomic JSON
writes retain the last recorded transition. Completed stages are reused only when
the composed plan, inputs and outputs match their recorded content signatures.
Manifest signatures include referenced image, mask and edge bytes. Sketch CSVs
also include their referenced masks. Large checkpoint signatures are streamed;
verification adds I/O rather than loading a second model into memory.

Training can resume from a saved epoch boundary. Sampling starts as a complete
stage and refuses to overwrite an existing generated manifest. A partial sampling
failure needs a fresh pipeline output directory. An interrupted evaluation can
be rerun; it writes its own result artifacts.

## Real-domain and generated-domain roles

The estimator trains on real training data. The generator trains on real defect
images and manually annotated masks. The recognizer mixes a quota-selected
generated pool with the real training split and receives teacher supervision on
both domains. Validation and testing use real samples only.

Audit checks guard image/group split overlap, generated holdouts, teacher split
identity and generator provenance. These checks are part of the service layer;
the generic engine remains usable by other research tasks.

## Controlled changes

Use a preset for a single structural change, and a sweep for comparisons:

| Scientific question | Configuration |
|:--|:--|
| Effect of an edge branch | `ablations/rgb_baseline.yaml` vs `tasks/estimator.yaml` |
| Effect of semantic supervision | `ablations/direct_augmentation.yaml` vs `tasks/recognizer.yaml` |
| Choice of fusion | `sweeps/fusion.yaml` |
| Backbone behavior | `sweeps/backbones.yaml` |
| Semantic temperature | `sweeps/temperature.yaml` |

The original utility `experiments` also emits component, ratio, temperature and
backbone ablations as flat execution configs. These use the same services.

Do not compare different class orders or independently changed dataset splits.
Use a fixed generated pool and frozen estimator when isolating recognizer choices.
For a full pipeline seed study, change each stage's seed through stage overrides
and choose a new pipeline output directory.

## Sweeps and reports

```yaml
schema_version: 1
base: ../tasks/recognizer.yaml
output: runs/sweeps/temperature
parameters:
  semantic.temperature: [5, 10, 15, 20]
  experiment.seed: [42, 43, 44]
```

The Cartesian product yields twelve runs. A descriptive run ID includes the
backbone, temperature, seed and a short configuration digest. Every run receives
its own directory, composed YAML, checkpoint and held-out prediction table.
Changing the sweep spec requires a new output directory.

```bash
pvaug sweep --config configs/sweeps/temperature.yaml --dry-run
pvaug sweep --config configs/sweeps/temperature.yaml
pvaug sweep --config configs/sweeps/temperature.yaml --resume
pvaug report --sweep runs/sweeps/temperature/sweep.json
```

The first failed job stops execution with its error recorded. Resume verifies
completed jobs and continues interrupted training from `last.pt`. The report
groups identical settings after excluding the seed field, then emits:

- `summary.json`: machine-readable means and sample standard deviations.
- `summary.csv`: a flat table suitable for subsequent plotting.
- `summary.md`: a reviewable table of accuracy and F1 results.

Metrics are fractions. A single run has a reported standard deviation of zero,
which does not estimate uncertainty. A sweep summary can include only completed
runs; inspect `sweep.json` for the planned/completed count before reporting it.

## CPU development path

```bash
pvaug demo-data --output data/demo
pvaug pipeline --config configs/debug/pipeline.yaml --dry-run
pvaug pipeline --config configs/debug/pipeline.yaml
```

Debug presets use tiny image/model dimensions and a random VQ first stage. They
are intended to check method wiring and artifacts. `pvaug smoke` generates the
same style of toy data inside a new run directory and constructs the complete
DAG automatically, without relying on repository-relative preset files.
