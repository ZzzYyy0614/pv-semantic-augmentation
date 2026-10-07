"""Create a source-only GitHub archive using explicit allowlists, never traversing data/.venv."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    ".gitignore",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
]
SOURCE_DIRS = [
    "src",
    "configs",
    "docs",
    "tests",
    "scripts",
    "assets",
    "reference",
    "third_party",
    ".github",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}


def package(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output == root or output.is_dir():
        raise ValueError("Output must be a zip file")
    candidates = [root / name for name in ROOT_FILES]
    for name in SOURCE_DIRS:
        candidates.extend(
            file
            for file in (root / name).rglob("*")
            if file.is_file()
            and not any(
                part in SKIP_PARTS or part.endswith(".egg-info")
                for part in file.relative_to(root).parts
            )
        )
    candidates = sorted(set(candidates))
    files = []
    for file in candidates:
        if file.suffix.lower() in {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".zip", ".safetensors"}:
            continue
        if file == output:
            continue
        relative = file.relative_to(root).as_posix()
        if file.stat().st_size > 2_000_000:
            raise ValueError(f"Unexpected large source file: {relative}")
        files.append((file, relative))
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = [
        {
            "path": relative,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "bytes": file.stat().st_size,
        }
        for file, relative in files
    ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file, relative in files:
            archive.write(file, "pv-semantic-edge/" + relative)
        archive.writestr("pv-semantic-edge/RELEASE_MANIFEST.json", json.dumps(inventory, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive verification failed")
    return {
        "archive": str(output),
        "files": len(files),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../pv-semantic-edge-github.zip")
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))
