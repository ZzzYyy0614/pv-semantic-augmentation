# Executed validation

Date: **2026-10-06**. Environment: Windows, Python 3.12.14, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, NumPy 2.5.2, Pillow 11.3.0. Torch CPU threads were bounded for the tests. No GPU training or real-dataset accuracy claim is made.

## Results

| Check | Result |
| --- | --- |
| Local editable package installation and installed `pvaug` entry point | Passed |
| Unit/integration suite | **48 passed** |
| Ruff checks on new source, tests and scripts | Passed |
| Complete CPU toy pipeline | Passed |
| ResNet18 / ConvNeXt recovered wrapper vs original forward | Identical logits within the test tolerance |
| ResNet18 / ConvNeXt-T / Swin-T paper variants | Forward and backward passed |
| Raw DDP-prefixed classifier checkpoint compatibility | Passed; incomplete weights rejected |
| Actual recovered `DGCF_Resnet18-100-regular.pth` | Strict load passed, 236 original state entries |
| Actual recovered large generator checkpoint metadata | Restricted weights-only read passed |
| Generator tensor structure against backup configuration | **623 tensors checked, zero missing, zero shape mismatches** |
| Legacy generator EMA mapping | 416 U-Net parameter buffers located |
| Class embedding in actual legacy generator | `(4, 512)`; no cross-attention transformer tensors |
| Class-conditioned paper generator | Class input changes the denoiser output; gradients reach class embedding |
| Backup generator class behavior | Class input is ignored, matching the recovered disabled-cross-attention configuration |
| DDIM sampling | Same input and seed give identical toy outputs |
| VQ first-stage gradients | Frozen during generator training |
| Optional reconstruction-only AE bootstrap | Trained a toy step, saved and reloaded the first stage |
| Data split checks | Pixel duplicates, group overlap, mask-size mismatch, generated holdouts and generator provenance leakage detected |
| Legacy annotation import | Preserved integer class order and mapped server paths to local files |
| Resume | Final model tensors match uninterrupted CPU training exactly in the controlled test |
| Shared engine | Classification, diffusion and VQ tasks execute the same trainer |
| Accumulation | Unequal microbatches and final partial groups match equivalent large-batch gradients in a model without BatchNorm |
| CPU bf16, clipping and AdamW | Executed successfully; callback order and optimizer-step limits checked |
| Frozen semantic task | Teacher parameters and BatchNorm state remain unchanged during student optimization |
| Diffusion resume | Final model and EMA tensors match uninterrupted CPU training exactly |
| Python / NumPy / Torch RNG | Snapshot and restore checked |
| Typed YAML composition | Defaults, child interpolation, nested overrides, type constraints and cycle/typo rejection checked |
| Backbone feature contract | ConvNeXt and Swin work under new registered names with identical predictions |
| Fusion extensions | Additive and concatenation forward/backward passed |
| Pipeline DAG | References, cycle rejection, failure recording, stage continuation and artifact-change rejection checked |
| Complete five-stage pipeline reuse | All completed stages reused after content verification |
| Multi-seed sweep | Two training/evaluation jobs executed, reused and aggregated successfully |
| Statistics | Mean and sample standard deviation checked against hand-computed values |
| Python wheel | v0.2.0 wheel built; module contents and license notices included |
| Local editable installation | Updated successfully to v0.2.0 |
| Framework diagram | SVG rendered to PNG and visually inspected |
| Experiment planner | Six component-ablation configs emitted successfully |
| Benchmark and confidence commands | Executed successfully on toy checkpoints |
| Grad-CAM | Overlay written and visually inspected |

## Toy pipeline scope

The latest complete smoke run used 20 toy images: 10 real-domain training examples, five validation examples and five test examples. The images imitate simple EL structure and defect geometry but are not physical EL observations.

A teacher was optimized for two steps; a tiny Sketch-LDM with a random first stage was optimized for two steps; DDIM generated two images for each of four defect labels (eight total); a student was optimized for two steps on the combined pool under the frozen teacher. Test evaluation, CSV probabilities and a Grad-CAM overlay were then produced.

The v0.2 smoke check uses the native ResNet path and the same artifact DAG API as
the public pipeline command. Its classifiers enable gradient clipping for the
tiny two-sample batches. The generated workflow configuration was then resumed;
all five completed stages were verified and reused. Toy metrics are omitted.

The toy accuracies are deliberately omitted here. They are software-test outputs and must not be used as paper reproduction results. Test output directories and toy checkpoints remain local under `runs/`, which is excluded from Git and source archives.

## Legacy generator scope

The inspected checkpoint is 6,417,969,036 bytes and records epoch 235. Its tensor shapes match the recovered backup architecture, whose U-Net contains 387,243,523 parameters. This comparison used a memory-mapped checkpoint and a meta-device model; it did not allocate and run the full denoiser, nor did it convert the entire checkpoint to a new artifact.

The verified small-model conversion route establishes the mapping logic; the large-model metadata check establishes names and dimensions. Full-size weight conversion, GPU synthesis, visual quality and downstream real-data metrics remain unverified. The original integer-to-class order and the original data split lists also remain to be connected.

## Re-run

```bash
python -m pytest -q
python -m ruff check src/pvaug tests scripts
python -m ruff format --check src tests scripts
pvaug smoke --output runs/new-smoke
pvaug pipeline --config configs/pipelines/paper.yaml --dry-run
pvaug sweep --config configs/sweeps/fusion.yaml --dry-run
```

Use a new smoke output path to preserve previous runs. The GitHub Actions workflow declares CPU validation on Python 3.10 and 3.12; those remote jobs have not been run locally or on GitHub.
