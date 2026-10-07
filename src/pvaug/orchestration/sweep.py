"""Cartesian experiment grids with independent run directories and multi-seed reports."""

from itertools import product
import csv
import json
from pathlib import Path
import statistics
from omegaconf import OmegaConf
from ..configuration import resolve_config, to_execution_config
from ..config import TrainConfig, config_dict
from .artifacts import atomic_json, fingerprint, timestamp, file_signature, manifest_signature


class Sweep:
    def __init__(self, base, parameters, output, root="."):
        self.base = Path(root) / base
        self.parameters, self.output = parameters, Path(output).resolve()
        if not parameters or any(
            not isinstance(values, list) or not values for values in parameters.values()
        ):
            raise ValueError("Sweep parameters must be non-empty lists")

    @classmethod
    def from_file(cls, path, output=None):
        path = Path(path).resolve()
        doc = resolve_config(path)
        if (
            set(doc) - {"schema_version", "base", "parameters", "output"}
            or doc.get("schema_version") != 1
        ):
            raise ValueError("Invalid sweep schema")
        return cls(doc["base"], doc["parameters"], output or doc["output"], path.parent)

    def plan(self):
        jobs = []
        keys = list(self.parameters)
        for values in product(*(self.parameters[key] for key in keys)):
            settings = dict(zip(keys, values))
            overrides = [f"{key}={json.dumps(value)}" for key, value in settings.items()]
            document = resolve_config(self.base, overrides)
            config = to_execution_config(document, TrainConfig)
            digest = fingerprint({"config": config_dict(config), "settings": settings})[:10]
            name = f"{config.model.backbone}-t{config.temperature:g}-s{config.seed}-{digest}"
            document.setdefault("experiment", {})["output"] = str(self.output / name)
            jobs.append({"id": name, "parameters": settings, "document": document})
        return jobs

    def run(self, resume=False):
        from ..train import train_classifier, evaluate_checkpoint

        jobs = self.plan()
        state_path = self.output / "sweep.json"
        digest = fingerprint(jobs)
        if state_path.exists():
            if not resume:
                raise FileExistsError("Sweep exists; use --resume or a new output")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["fingerprint"] != digest:
                raise ValueError("Sweep configuration changed")
        else:
            state = {"fingerprint": digest, "created_at": timestamp(), "jobs": {}}
        for job in jobs:
            config = to_execution_config(job["document"], TrainConfig)
            folder = Path(config.output)
            inputs = {"manifest": manifest_signature(config.manifest)}
            if config.generated_manifest:
                inputs["generated_manifest"] = manifest_signature(config.generated_manifest)
            if config.teacher:
                inputs["teacher"] = file_signature(config.teacher)
            entry = state["jobs"].get(job["id"])
            if entry and entry["status"] == "completed":
                checkpoint = folder / "best.pt"
                metrics_path = folder / "test" / "metrics.json"
                if not checkpoint.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(f"Completed sweep artifacts missing: {job['id']}")
                artifacts = {
                    "checkpoint": file_signature(checkpoint),
                    "metrics": file_signature(metrics_path),
                }
                if entry["inputs"] != inputs or entry["artifacts"] != artifacts:
                    raise ValueError(f"Completed sweep inputs or artifacts changed: {job['id']}")
                continue
            if entry and entry.get("inputs") != inputs:
                raise ValueError(f"Interrupted sweep inputs changed: {job['id']}")
            folder.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(job["document"]), folder / "resolved.yaml")
            entry = {
                "status": "running",
                "parameters": job["parameters"],
                "started_at": timestamp(),
                "inputs": inputs,
            }
            state["jobs"][job["id"]] = entry
            atomic_json(state_path, state)
            try:
                last = folder / "last.pt"
                checkpoint = train_classifier(
                    config, resume=last if resume and last.exists() else None
                )
                metrics = evaluate_checkpoint(
                    checkpoint,
                    config.manifest,
                    folder / "test",
                    device=config.device,
                    batch_size=config.batch_size,
                )
                entry.update(
                    status="completed",
                    metrics=metrics,
                    output=str(folder),
                    completed_at=timestamp(),
                    artifacts={
                        "checkpoint": file_signature(checkpoint),
                        "metrics": file_signature(folder / "test" / "metrics.json"),
                    },
                )
            except Exception as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                atomic_json(state_path, state)
                raise
            atomic_json(state_path, state)
        summarize_sweep(state_path)
        return state


def summarize_sweep(path):
    path = Path(path)
    state = json.loads(path.read_text(encoding="utf-8"))
    groups = {}
    for entry in state["jobs"].values():
        if entry["status"] != "completed":
            continue
        parameters = {
            key: value
            for key, value in entry["parameters"].items()
            if key not in {"experiment.seed", "seed"}
        }
        key = json.dumps(parameters, sort_keys=True)
        groups.setdefault(key, []).append(entry)
    rows = []
    for key, entries in groups.items():
        row = {"setting": key, "runs": len(entries)}
        for metric in ["accuracy", "f1_macro", "f1_weighted"]:
            values = [entry["metrics"][metric] for entry in entries]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    atomic_json(path.parent / "summary.json", rows)
    if rows:
        with (path.parent / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Experiment summary",
        "",
        "Sample standard deviation across seeds; fractions, not percentages.",
        "Single-run standard deviation is reported as 0; it does not estimate uncertainty.",
        "",
        "| Setting | Runs | Accuracy | Macro F1 | Weighted F1 |",
        "|:--|--:|--:|--:|--:|",
    ]
    for row in rows:
        values = [
            f"{row[metric + '_mean']:.4f} ± {row[metric + '_std']:.4f}"
            for metric in ["accuracy", "f1_macro", "f1_weighted"]
        ]
        lines.append("| " + " | ".join([row["setting"], str(row["runs"]), *values]) + " |")
    (path.parent / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
