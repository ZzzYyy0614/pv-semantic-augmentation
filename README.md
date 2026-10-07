# PV-Semantic-Augmentation

Code for **Semantic-aware data augmentation with edge priors for PV module defect recognition using EL images**.

Yu Zhu, Yuan Cao, Yuxuan Wu and Qiang Yang · *Solar Energy* 305 (2026), 114269 · [Paper](https://doi.org/10.1016/j.solener.2025.114269)

![Method and code organization](assets/framework.svg)

[中文说明](docs/README_zh.md) · [Implementation](docs/architecture.md) · [Experiments](docs/experiments.md) · [Reproduction notes](docs/reproduction.md)

## Method

The method combines sketch-conditioned image generation with edge-aware defect recognition:

1. **Sketch-LDM** generates defect images from region sketches and class labels. Image and sketch latents are encoded by a frozen VQ first stage; the sketch latent is concatenated with the noisy image latent.
2. **Semantic estimator** learns class probabilities from the real training images. Its parameters remain frozen during recognizer training.
3. **Defect recognizer** combines visual features with a Sobel edge branch at three scales. It is trained on real and generated images using cross-entropy and temperature-scaled KL supervision from the estimator.

The recognition models include ResNet18, ConvNeXt-Tiny and Swin-T. Gated fusion is the default. Additive and concatenation fusion are included for further ablation studies.

## Installation

Use Python 3.10–3.12, PyTorch 2.5.x and torchvision 0.20.x. For a CPU environment:

```bash
python -m venv .venv
# Activate the environment before installing.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

For GPU training, install the corresponding CUDA build of PyTorch first. Commands below can also be invoked with `python -m pvaug`.

A small CPU example checks training, generation, evaluation and Grad-CAM without external data or weights:

```bash
pvaug smoke --output runs/example
```

This example uses synthetic test images and a randomly initialized small VQ encoder. Its metrics are not reproduction results.

## Training

Dataset and trained weights are not included. Dataset release information will be added separately. The data interface uses CSV columns:

```text
image,label,split,domain,mask,edge,group
```

Image paths are relative to the CSV. Generator training requires annotated region masks. See [experiment settings](docs/experiments.md) for the roles of the real and generated splits.

The main configurations are:

| Configuration | Use |
|:--|:--|
| `configs/tasks/estimator.yaml` | Train the semantic estimator on real images |
| `configs/tasks/sketch_ldm.yaml` | Train the sketch-conditioned generator |
| `configs/tasks/recognizer.yaml` | Train the recognizer with generated images and semantic supervision |
| `configs/pipelines/paper.yaml` | Run the five stages, including generation and evaluation |

Set the manifest, class order, sketches and VQ checkpoint before running the full experiment. Inspect the resolved settings first:

```bash
pvaug config --config configs/tasks/estimator.yaml
pvaug pipeline --config configs/pipelines/paper.yaml --dry-run
```

Run the full experiment or train a single model:

```bash
pvaug pipeline --config configs/pipelines/paper.yaml
pvaug run --config configs/tasks/estimator.yaml
```

Configurations inherit shared settings from `configs/base/`. Individual parameters can be overridden without copying the full configuration:

```bash
pvaug run --config configs/tasks/recognizer.yaml --set model.backbone=convnext_tiny --set experiment.output=runs/convnext
```

Training writes the configuration, history and checkpoints to the selected run directory. To continue a saved run:

```bash
pvaug run --config configs/tasks/estimator.yaml --resume runs/estimator/last.pt
pvaug pipeline --config configs/pipelines/paper.yaml --resume
```

Training resumes at an epoch boundary. The pipeline checks saved stage inputs and outputs before reusing them. See [experiments.md](docs/experiments.md) for recovery details and limitations.

## Evaluation and ablations

```bash
pvaug eval --checkpoint runs/paper/recognizer/best.pt --manifest data/real.csv --output runs/test
pvaug predict --checkpoint runs/paper/recognizer/best.pt --input data/example.png --output runs/predict --gradcam
```

Evaluation reports accuracy, macro precision/recall/F1, weighted F1, per-class results and a confusion matrix. Predictions are also saved as per-image probabilities.

The sweep configurations cover backbone, fusion and temperature comparisons across random seeds:

```bash
pvaug sweep --config configs/sweeps/fusion.yaml --dry-run
pvaug sweep --config configs/sweeps/fusion.yaml
pvaug report --sweep runs/sweeps/fusion/sweep.json
```

Reports contain the mean and sample standard deviation across completed runs. Additional fusion strategies are experimental options; their results are not reported in the paper.

## Code layout

```text
src/pvaug/
├── models/          # Recognition networks, Sketch-LDM and backbone adapters
├── tasks/           # Classification, semantic supervision and generation losses
├── engine/          # Training loop, callbacks and checkpoints
├── configuration/   # YAML loading and validation
├── orchestration/   # Multi-stage runs and parameter sweeps
├── data/            # Manifests, transforms and sampling
├── train.py         # Classifier training and evaluation
├── generation.py    # Generator training and sampling
└── inference.py     # Prediction, Grad-CAM and timing
configs/             # Training settings and experiment variants
reference/           # Earlier model and training source
third_party/         # Source attribution and licenses
```

Classification, diffusion and VQ training share the same training loop. New losses belong in `tasks/`; backbone and fusion additions belong in `models/`. [Implementation notes](docs/architecture.md) provide examples of these interfaces.

## Model versions and reproduction

`variant: paper` uses the common Sigmoid fusion design and torchvision backbones. `variant: recovered` preserves the earlier classifier parameter names and execution paths for loading existing weights. The generator compatibility configuration uses L1 and disables class cross-attention; the paper configuration uses MSE with class conditioning enabled.

These settings require different weights. Other differences between the paper description and earlier source are documented in [reproduction.md](docs/reproduction.md). Flat configurations are retained in `configs/legacy/`; `pvaug legacy --help` lists the utility commands.

Local CPU checks cover model interfaces, losses, checkpoint loading and recovery. Real-data accuracy and GPU performance have not been validated in this release. Details are recorded in [verification.md](docs/verification.md).

## Citation

```bibtex
@article{zhu2026semantic,
  title = {Semantic-aware data augmentation with edge priors for PV module defect recognition using EL images},
  author = {Zhu, Yu and Cao, Yuan and Wu, Yuxuan and Yang, Qiang},
  journal = {Solar Energy},
  volume = {305},
  pages = {114269},
  year = {2026},
  doi = {10.1016/j.solener.2025.114269}
}
```

## License

See [LICENSE](LICENSE) for the new project code and [third_party/NOTICE.md](third_party/NOTICE.md) for imported source and its license status.
