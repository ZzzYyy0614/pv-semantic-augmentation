# Source provenance and notices

- `src/pvaug/vendor/ldm` contains the necessary encoder, decoder, U-Net, attention and utilities recovered from the author's CompVis latent-diffusion checkout. Absolute imports were namespaced under `pvaug.vendor`. The CompVis MIT notice is preserved in `CompVis-LDM-LICENSE`. Upstream: https://github.com/CompVis/latent-diffusion.
- `src/pvaug/vendor/quantize.py` is the recovered taming-transformers vector quantizer. Its MIT notice is in `Taming-Transformers-LICENSE`. Upstream: https://github.com/CompVis/taming-transformers.
- `src/pvaug/models/recovered` and `reference/classifier` preserve the author's classifier code, including the ResNet base and training utilities attributed to Baiyu in the backup. The originating base project is https://github.com/weiaicunzai/pytorch-cifar100. No separate license file was present in that classifier backup; these files are retained with their original author notices, rather than assigned the new interface's MIT license.
- `reference/generator` is a selected, unmodified snapshot of the author's generator code/configuration, with the original CompVis license. It contains historical absolute server paths for traceability and is not a runnable package entry point.
- PyTorch and torchvision are dependencies, not bundled source distributions. The paper-described backbone option uses their installed implementations.

`docs/source_inventory.json` identifies each imported file, its origin, its SHA-256 hash, and any import-only modification. It excludes source datasets, model weights, cached downloads, and training logs. This inventory and the retained notices should remain with a source release. Dataset and model redistribution require their own terms.
