"""One-time source import used while preparing this release; no datasets or weights copied."""

import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(
    description="Import selected author backup source, excluding data/weights"
)
parser.add_argument("--classifier", required=True)
parser.add_argument("--generator", required=True)
args = parser.parse_args()
CLASSIFIER = Path(args.classifier).resolve()
GENERATOR = Path(args.generator).resolve()
records = []


def copy(source, target, origin, transform=None):
    target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    if transform:
        data = transform(data.decode("utf-8")).encode("utf-8")
    target.write_bytes(data)
    records.append(
        {
            "origin": origin,
            "destination": target.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(data).hexdigest(),
            "change": "namespace imports" if transform else "unchanged snapshot",
        }
    )


for name in [
    "models/resnet.py",
    "models/convnext.py",
    "models/swinT_small.py",
    "train.py",
    "train_KD.py",
    "utils.py",
    "data/edge_sobel.py",
    "conf/global_settings.py",
]:
    copy(CLASSIFIER / name, "reference/classifier/" + name, "pytorch-baseline/" + name)
for name in ["models/resnet.py", "models/convnext.py"]:
    copy(
        CLASSIFIER / name,
        "src/pvaug/models/recovered/" + Path(name).name,
        "pytorch-baseline/" + name,
    )
for name in [
    "ldm/models/diffusion/ddpm.py",
    "ldm/models/autoencoder.py",
    "ldm/data/juju_dataset.py",
    "configs/latent-diffusion/ju_multi_condition.yaml",
    "configs/autoencoder/VQ_VAE.yaml",
    "ju_multicondition_generate.py",
    "environment.yaml",
    "requirement.txt",
    "LICENSE",
]:
    copy(GENERATOR / name, "reference/generator/" + name, "latent-diffusion/" + name)
for name in [
    "ldm/modules/diffusionmodules/openaimodel.py",
    "ldm/modules/diffusionmodules/util.py",
    "ldm/modules/diffusionmodules/model.py",
    "ldm/modules/attention.py",
    "ldm/util.py",
]:
    copy(
        GENERATOR / name,
        "src/pvaug/vendor/" + name,
        "latent-diffusion/" + name,
        lambda text: text.replace("from ldm.", "from pvaug.vendor.ldm."),
    )
copy(
    GENERATOR / "src/taming-transformers/taming/modules/vqvae/quantize.py",
    "src/pvaug/vendor/quantize.py",
    "taming-transformers/taming/modules/vqvae/quantize.py",
)
copy(GENERATOR / "LICENSE", "third_party/CompVis-LDM-LICENSE", "latent-diffusion/LICENSE")
for folder in (ROOT / "src/pvaug/vendor").rglob("*"):
    if folder.is_dir():
        (folder / "__init__.py").touch()
(ROOT / "src/pvaug/vendor/__init__.py").touch()
(ROOT / "src/pvaug/models/recovered/__init__.py").touch()
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/source_inventory.json").write_text(
    json.dumps(records, indent=2) + "\n", encoding="utf-8"
)
print(f"Imported {len(records)} source files (no images, checkpoints, or experiment logs).")
