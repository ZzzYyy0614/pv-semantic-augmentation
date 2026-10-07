"""Durable pipeline state and content signatures for artifact reuse."""

import hashlib
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from ..utils import sha256


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_signature(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def manifest_signature(path):
    """Manifest identity includes referenced image/mask/edge bytes, not just the CSV."""
    path = Path(path).resolve()
    signatures = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in ["image", "mask", "edge"]:
                if row.get(key):
                    signatures.append(file_signature(path.parent / row[key]))
    return {
        "manifest": file_signature(path),
        "referenced_files": len(signatures),
        "content_sha256": fingerprint(signatures),
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
