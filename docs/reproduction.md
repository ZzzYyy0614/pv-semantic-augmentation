# Method fidelity and implementation choices

This document records the correspondence between the paper, the earlier training source and the current implementation. Models from `pytorch-baseline` and `latent-diffusion` are retained where needed to load existing weights. The default experiment settings and changes to those implementations are listed below.

## Paper-to-code mapping

| Paper component | Active implementation | Provenance / decision |
| --- | --- | --- |
| Sobel edge magnitude, Section 3.4 | `models/edge.py`, `data/dataset.py` | Fixed horizontal/vertical kernels, per-image maximum normalization, reflect padding matching OpenCV's border convention. Blank images have a zero map. Cached original edge images can be supplied. |
| Edge-guided ResNet18, Fig. 6 | `models/recovered/resnet.py`, wrapper in `models/classifier.py` | Retains `DGCF_Res18` and its exact forward route. Three gates use 3×3 convolutions and Softmax. |
| Paper GFB | `models/edge.py`, `variant=paper` | Reconstructed Sigmoid gates with two spatial maps and 1×1 convolutions. The wrapper adds their fused output back to the backbone, interpreting the outer plus nodes in Fig. 6 as residual additions. Gate widths and precise edge stem are not specified in the paper. |
| ConvNeXt | `models/recovered/convnext.py` | Recovered four-stage convolutional alignment/fusion differs from the common GFB in Fig. 6. Preserved under `variant=recovered`. |
| ConvNeXt-T / Swin-T paper option | `models/classifier.py`, `variant=paper` | Installed torchvision backbones; three new Sigmoid fusion points. This option is not binary compatible with recovered checkpoints. |
| Sketch-LDM encoder/decoder | `models/sketch_ldm.py`, vendored CompVis encoder/decoder/quantizer | Recovered VQModelInterface: three latent channels, factor-four reduction, 256-entry codebook, pre-quantization encode and quantized decode. |
| Sketch spatial condition | `SketchLDM.encode` | The frozen first stage encodes the sketch; its three latent channels concatenate with the three noisy image-latent channels. No ControlNet or text prompt is introduced. |
| Class condition | `SketchLDM.predict_noise` | One learned class token, 512 dimensions by default. Paper mode enables the recovered U-Net's spatial transformer/cross-attention. |
| Noise loss, Eq. (1) | `SketchLDM.loss` | Paper mode uses MSE; backup mode uses recovered L1. Both predict injected noise at uniformly sampled time steps. |
| Sampling | `SketchLDM.sample` | Native epsilon-prediction DDIM with configurable steps, eta, and seed. The new time grid spans the full schedule; it does not reproduce the old sampler's `+1` indexing convention. |
| Frozen semantic estimator | `tasks/classification.py`, assembly in `train.py` | Eval mode and disabled gradients, including BatchNorm state. Teacher logits supervise real and generated images. |
| Semantic objective, Eq. (2) | `losses.py` | Batch-mean KL(teacher || student), multiplied by T², plus cross entropy. Uses logits directly and does not apply softmax twice. |
| Alpha annealing | `losses.py` | Recovered code uses `max(0.7-0.4*epoch/100,0.3)` over epochs 1–100. New zero-based schedule uses exact endpoints 0.7 and 0.3; this slight endpoint change is explicit. |
| Experiments and ablations | `orchestration/`, `experiments.py` | Artifact-based method stages and configuration grids; configurable seed, model, temperature, ratio and teacher. Test set remains real and separate. |

## Framework execution changes in v0.2

All method tasks use the same trainer. The default fp32, accumulation-one execution
preserves the method objectives; optional clipping, precision and optimizer changes
are explicit experiment settings. Gradient accumulation normalizes by the actual
number of samples, including final partial groups. It does not reproduce the
BatchNorm statistics of a larger physical batch. Debug presets enable clipping to
stabilize the tiny two-sample CPU checks; paper presets leave it disabled.

Native classifiers now honor a supplied edge tensor, consistent with the recovered
classifier path. The dataset computes Sobel on the original image and quantizes to
uint8 levels before resizing, or reads a cached edge. A direct model call without
an edge computes it from its input tensor. The v0.1 native wrapper always recomputed
the prior, so native checkpoint tensor compatibility does not imply identical
preprocessing to that initial interface.

Additive and concatenation fusion are framework extensions for additional ablation
studies. They are not experiments claimed by the published paper. The default
gated fusion retains the documented interpretation above.

## Material differences found in the backup

1. **Classifier architecture and parameter counts.** The recovered ResNet stem is a 3×3 stride-one convolution without a max-pool, rather than torchvision's 7×7 stride-two stem. `DGCF_Res18` also registers unused alignment modules and an unused edge block. Those tensors remain present for exact weight compatibility. The recovered five-class edge model contains **13,101,175 parameters**. This agrees approximately with Table 7's 13.1M but not Tables 3/5's 12.1M. The paper's 11.7M baseline count also does not describe this exact five-class backup. Measured counts are reported separately from published counts.
2. **Fusion activation.** The paper describes Sigmoid gates. The recovered ResNet's executed gate network uses Softmax. Its RGB/edge alignment modules are defined but not called. The wrapper preserves that execution route; paper mode provides an explicit separate interpretation.
3. **Generated class control.** `ju_multi_condition.yaml` and the saved October 2024 configuration pass the class embedding as `context` but omit `use_spatial_transformer`. Its default is false, so the executed AttentionBlock ignores context. The inspected generator checkpoint contains a `(4,512)` class embedding and no transformer-block tensors. Consequently that backup cannot be assumed to implement the class control claimed in the paper. Enabling it changes the model tensor set and requires training rather than a permissive `strict=False` load.
4. **Diffusion loss and schedule.** The saved configuration uses L1, 1000 diffusion steps, beta endpoints 0.0015 and 0.0205, and a three-channel VQ first stage. Paper mode changes L1 to MSE and enables cross-attention while retaining the recovered architecture dimensions. The first-stage source config additionally uses LPIPS/discriminator training; the portable optional AE bootstrap deliberately uses reconstruction + codebook loss instead.
5. **Training code consistency.** One recovered KD loop unpacks two outputs while its selected ResNet returns a single logit tensor. Legacy paths, device selection, wandb logging, and several experimental alternatives are hardcoded. New loops use a consistent logit interface, one configurable device, portable paths, and local JSON logs.
6. **Swin shape errors.** The archived Swin stage passes `[B,L,C]` tokens into a patch-merging implementation expecting `[B,H,W,C]`, and window division does not pad arbitrary later spatial sizes. The source is retained in `reference/classifier`; the runnable Swin option uses torchvision's implementation. Its checkpoints are not presented as recovered Swin weights.
7. **Optimizer settings.** Table 2 gives SGD weight decay 0.05; the recovered KD script uses 0.1. Default configs follow Table 2 (100 epochs, batch 24, LR 0.0005, momentum 0.9, decay 0.05); generator defaults follow 200 epochs, batch 4, Adam LR 0.000001. The paper does not specify a classifier LR scheduler, so the default is constant, with cosine available as an explicit alternative. The old generator warmup/cosine schedule is not restored in the portable trainer.

## Dataset and model-selection ambiguity

The paper reports 2100 images, four defect classes with 300 images each, 100 held-out images per defect class, and 200 generated images per defect class. The four classes account for 1200 images; the defect-free class count and its split are not specified completely. Exact original train/test lists are required to resolve the remaining counts. The project does not invent them.

The paper describes selecting models on test performance. This release selects checkpoints using validation, keeps the test split out of optimization/model selection, and reports the protocol change. If only the original training/test folders are supplied, validation is carved from training; this reduces the optimization pool relative to the published training counts. To use all recovered training images, set `--val-fraction 0` and accept training-loss checkpoint selection, or provide a separate appropriate validation dataset. Exact published results are still not guaranteed.

Original classifier label IDs come from sorted folder names. Original generator IDs come from integer annotation labels; they need not match. Supplied default class lists are a new canonical convention. Before using old weights, recover the actual two orders and use explicit configs. Raw `.pth` files do not contain enough information to infer semantic class names safely.

Generated provenance from this project is checked against the classifier's real training hashes. External generated datasets and converted legacy weights may have unknown provenance; exact pixel/group audits do not prove that a generator or first stage never saw test data. Inspect the original annotation lists and checkpoint training history when conducting a scientific reproduction.

## What is verified

See `verification.md` for executed tests. Unit tests probe objective direction and gradient ownership, minority-class metrics, recovered forward parity, strict checkpoint compatibility, paper-backbone gradients, class conditioning, deterministic DDIM sampling, data leakage, and resume equivalence. A full CPU toy-data pipeline covers training, synthesis, evaluation, inference, and Grad-CAM.

These tests establish software behavior. They do not establish visual fidelity on real EL images, the paper's 91.6% result, its average gain, its measured FPS/FLOPs, or CUDA performance. Large legacy generator tensor metadata was inspected without a full-scale conversion or inference run. The original generator checkpoint may require several GB of memory and disk space.

## Source references

- Author-provided paper: [DOI](https://doi.org/10.1016/j.solener.2025.114269).
- [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion), including the recovered encoder, diffusion U-Net and attention code.
- [CompVis taming-transformers](https://github.com/CompVis/taming-transformers), vector quantizer and first-stage conventions.
- [torchvision 0.20 Swin implementation](https://docs.pytorch.org/vision/0.20/models/swin_transformer.html), used by the portable paper-described variant.

Historical source snapshots are reference material, not commands or configuration instructions for the new project.
