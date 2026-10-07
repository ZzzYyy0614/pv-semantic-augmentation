"""A typed artifact DAG for the paper's estimator -> synthesis -> recognition workflow."""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
from ..configuration import resolve_config, to_execution_config
from ..config import config_dict, TrainConfig, LDMConfig
from .artifacts import fingerprint, file_signature, manifest_signature, timestamp, atomic_json

TRAINING = {"classification", "diffusion", "autoencoder"}
KINDS = TRAINING | {"sampling", "evaluation"}
OUTPUTS = {
    "classification": {"checkpoint": "best.pt"},
    "diffusion": {"checkpoint": "last.pt"},
    "autoencoder": {"checkpoint": "first_stage.pt"},
    "sampling": {"manifest": "generated.csv"},
    "evaluation": {"metrics": "metrics.json", "predictions": "predictions.csv"},
}


@dataclass(frozen=True)
class Stage:
    id: str
    kind: str
    config: str | None = None
    depends_on: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.id):
            raise ValueError("Stage IDs must be lowercase names without path separators")
        if self.kind not in KINDS:
            raise ValueError(f"Unknown stage kind: {self.kind}")
        if self.kind in TRAINING and not self.config:
            raise ValueError(f"Training stage {self.id} requires a config")
        if any(not isinstance(value, str) for value in self.inputs.values()):
            raise ValueError("Stage input values must be paths or artifact references")
        if self.kind not in TRAINING:
            required = {
                "sampling": {"checkpoint", "sketches"},
                "evaluation": {"checkpoint", "manifest"},
            }[self.kind]
            if set(self.inputs) != required:
                raise ValueError(f"{self.kind} requires inputs {sorted(required)}")
            allowed = {
                "sampling": {"per_class", "steps", "seed", "device", "eta"},
                "evaluation": {"split", "device", "batch_size"},
            }[self.kind]
            if set(self.options) - allowed:
                raise ValueError(f"Unknown options for {self.kind}")
        elif self.options:
            raise ValueError("Training stages use config overrides, not options")

    @property
    def dependencies(self):
        references = {
            value[1:].split(".")[0] for value in self.inputs.values() if value.startswith("@")
        }
        return set(self.depends_on) | references


class Pipeline:
    def __init__(self, name, output, stages, config_root="."):
        self.name, self.output = name, Path(output).resolve()
        self.root = Path(config_root).resolve()
        self.stages = stages
        self.by_id = {stage.id: stage for stage in stages}
        if len(self.by_id) != len(stages):
            raise ValueError("Duplicate pipeline stage ID")
        self.order = self.topological_order()

    @classmethod
    def from_file(cls, path, output=None, overrides=()):
        path = Path(path).resolve()
        doc = resolve_config(path, overrides)
        unknown = set(doc) - {"schema_version", "name", "output", "stages"}
        if unknown or doc.get("schema_version") != 1:
            raise ValueError(f"Invalid pipeline schema: {sorted(unknown)}")
        return cls(
            doc["name"],
            output or doc["output"],
            [Stage(**stage) for stage in doc["stages"]],
            path.parent,
        )

    def topological_order(self):
        pending = {stage.id: set(stage.dependencies) for stage in self.stages}
        unknown = set().union(*pending.values()) - set(pending) if pending else set()
        if unknown:
            raise ValueError(f"Unknown pipeline dependencies: {sorted(unknown)}")
        order = []
        while pending:
            ready = [key for key, dependencies in pending.items() if not dependencies]
            if not ready:
                raise ValueError(f"Pipeline dependency cycle among {sorted(pending)}")
            for key in ready:
                order.append(key)
                pending.pop(key)
                for dependencies in pending.values():
                    dependencies.discard(key)
        # Validate references before producing any artifacts.
        for stage in self.stages:
            for value in stage.inputs.values():
                if value.startswith("@"):
                    parts = value[1:].split(".")
                    if len(parts) != 2 or parts[1] not in OUTPUTS[self.by_id[parts[0]].kind]:
                        raise ValueError(f"Unknown artifact reference: {value}")
        return order

    def outputs(self, stage):
        return {
            key: str(self.output / stage.id / filename)
            for key, filename in OUTPUTS[stage.kind].items()
        }

    def inputs(self, stage):
        resolved = {}
        for key, value in stage.inputs.items():
            if value.startswith("@"):
                source, artifact = value[1:].split(".")
                resolved[key] = self.outputs(self.by_id[source])[artifact]
            else:
                # Dataset/weight paths are workspace-relative; configs are file-relative.
                resolved[key] = str(Path(value).resolve())
        return resolved

    def training_config(self, stage):
        doc = resolve_config(self.root / stage.config, stage.overrides)
        if "task" in doc and doc["task"] != stage.kind:
            raise ValueError(f"Stage {stage.id} kind differs from config task")
        config = to_execution_config(
            doc, TrainConfig if stage.kind == "classification" else LDMConfig
        )
        inputs = self.inputs(stage)
        allowed = {
            "classification": {"teacher", "manifest", "generated_manifest"},
            "diffusion": {"manifest", "first_stage_checkpoint", "initialize"},
            "autoencoder": {"manifest"},
        }[stage.kind]
        if set(inputs) - allowed:
            raise ValueError(f"Unsupported inputs for {stage.id}: {set(inputs) - allowed}")
        return replace(config, output=str(self.output / stage.id), **inputs)

    def plan(self):
        return {
            "name": self.name,
            "output": str(self.output),
            "stages": [
                {
                    "id": stage.id,
                    "kind": stage.kind,
                    "depends_on": sorted(stage.dependencies),
                    "inputs": self.inputs(stage),
                    "outputs": self.outputs(stage),
                    "config": config_dict(self.training_config(stage))
                    if stage.kind in TRAINING
                    else None,
                    "options": stage.options,
                }
                for stage in (self.by_id[key] for key in self.order)
            ],
        }

    def execute_stage(self, stage, resume=False):
        output = self.output / stage.id
        inputs = self.inputs(stage)
        if stage.kind in TRAINING:
            from ..train import train_classifier
            from ..generation import train_generator, train_autoencoder

            functions = {
                "classification": train_classifier,
                "diffusion": train_generator,
                "autoencoder": train_autoencoder,
            }
            checkpoint = output / "last.pt"
            functions[stage.kind](
                self.training_config(stage),
                resume=checkpoint if resume and checkpoint.exists() else None,
            )
        elif stage.kind == "sampling":
            from ..generation import generate_samples

            generate_samples(inputs["checkpoint"], inputs["sketches"], output, **stage.options)
        elif stage.kind == "evaluation":
            from ..train import evaluate_checkpoint

            evaluate_checkpoint(inputs["checkpoint"], inputs["manifest"], output, **stage.options)

    def run(self, resume=False):
        plan = self.plan()  # Full schema validation precedes execution.
        state_path = self.output / "pipeline.json"
        digest = fingerprint(plan)
        if state_path.exists():
            if not resume:
                raise FileExistsError("Pipeline state exists; use --resume or a new output")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["fingerprint"] != digest:
                raise ValueError("Pipeline configuration changed; choose a new output")
        else:
            state = {
                "name": self.name,
                "fingerprint": digest,
                "created_at": timestamp(),
                "stages": {},
                "plan": plan,
            }
        atomic_json(self.output / "plan.json", plan)
        for description in plan["stages"]:
            stage = self.by_id[description["id"]]
            input_signatures = {
                key: (
                    manifest_signature(path)
                    if key in {"manifest", "generated_manifest", "sketches"}
                    else file_signature(path)
                )
                for key, path in self.inputs(stage).items()
            }
            # Track training data and external first-stage/teacher files as well as explicit inputs.
            if stage.kind in TRAINING:
                config = self.training_config(stage)
                for key in [
                    "manifest",
                    "teacher",
                    "generated_manifest",
                    "first_stage_checkpoint",
                    "initialize",
                ]:
                    path = getattr(config, key, None)
                    if path:
                        input_signatures[key] = (
                            manifest_signature(path)
                            if key in {"manifest", "generated_manifest"}
                            else file_signature(path)
                        )
            entry = state["stages"].get(stage.id)
            if entry and entry["status"] == "completed":
                signatures = {
                    key: file_signature(path) for key, path in self.outputs(stage).items()
                }
                if entry["inputs"] != input_signatures or entry["artifacts"] != signatures:
                    raise ValueError(
                        f"Artifacts or inputs changed for {stage.id}; choose a new output"
                    )
                print(f"[pipeline] reuse {stage.id}", flush=True)
                continue
            if entry and entry.get("inputs") != input_signatures:
                raise ValueError(f"Inputs changed for interrupted stage {stage.id}")
            entry = {"status": "running", "started_at": timestamp(), "inputs": input_signatures}
            state["stages"][stage.id] = entry
            atomic_json(state_path, state)
            print(f"[pipeline] start {stage.id} ({stage.kind})", flush=True)
            try:
                self.execute_stage(stage, resume)
                entry.update(
                    status="completed",
                    completed_at=timestamp(),
                    artifacts={
                        key: file_signature(path) for key, path in self.outputs(stage).items()
                    },
                )
            except Exception as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                atomic_json(state_path, state)
                raise
            atomic_json(state_path, state)
        state["completed_at"] = timestamp()
        atomic_json(state_path, state)
        return state
