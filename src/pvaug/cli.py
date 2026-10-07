"""Public framework commands. Original flat commands remain in legacy_cli.py."""

import argparse
import json
from pathlib import Path
import sys
import torch
from omegaconf import OmegaConf
from .configuration import resolve_config, to_execution_config

LEGACY_COMMANDS = {
    "prepare",
    "audit",
    "import-legacy",
    "demo-data",
    "train",
    "train-generator",
    "train-autoencoder",
    "eval",
    "predict",
    "generate",
    "convert-generator",
    "convert-classifier",
    "benchmark",
    "confidence",
    "experiments",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pvaug", description="Semantic-aware EL research framework"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--version", action="version", version="pvaug 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train a typed classification, diffusion or VQ task")
    run.add_argument("--config", required=True)
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--resume")
    run.add_argument("--init", help="Classifier initialization checkpoint")
    config = commands.add_parser("config", help="Compose, resolve and validate a configuration")
    config.add_argument("--config", required=True)
    config.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    config.add_argument("--format", choices=["yaml", "json"], default="yaml")
    config.add_argument("--output")
    pipeline = commands.add_parser("pipeline", help="Inspect or execute an artifact DAG")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    pipeline.add_argument("--output")
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.add_argument("--resume", action="store_true")
    sweep = commands.add_parser("sweep", help="Plan or run a parameter grid across random seeds")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--output")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--resume", action="store_true")
    report = commands.add_parser("report", help="Aggregate sweep results as CSV, JSON and Markdown")
    report.add_argument("--sweep", required=True, help="Path to sweep.json")
    commands.add_parser("components", help="List registered model components")
    smoke = commands.add_parser("smoke", help="Exercise the complete method on CPU toy images")
    smoke.add_argument("--output", default="runs/smoke")
    commands.add_parser("legacy", help="Show the original utility commands (pvaug legacy --help)")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # --threads is the only global option taking a value. Preserve old direct invocations.
    tokens = argv[2:] if argv[:1] == ["--threads"] else argv
    if tokens and (tokens[0] in LEGACY_COMMANDS or tokens[0] == "legacy"):
        from .legacy_cli import main as legacy_main

        forwarded = argv.copy()
        if tokens[0] == "legacy":
            forwarded.remove("legacy")
        return legacy_main(forwarded)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command in {"run", "config"}:
        document = resolve_config(args.config, args.set)
        config = to_execution_config(document)
        if args.command == "config":
            payload = (
                OmegaConf.to_yaml(OmegaConf.create(document))
                if args.format == "yaml"
                else json.dumps(document, indent=2, ensure_ascii=False)
            )
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload + "\n", encoding="utf-8")
            print(payload)
            return document
        from .train import train_classifier
        from .generation import train_generator, train_autoencoder

        task = document.get("task", "classification")
        if args.init and task != "classification":
            parser.error(
                "--init is for classifiers; diffusion initialization is in model.initialization"
            )
        if task == "classification":
            result = train_classifier(config, args.resume, args.init)
        else:
            function = train_generator if task == "diffusion" else train_autoencoder
            result = function(config, args.resume)
        OmegaConf.save(OmegaConf.create(document), Path(config.output) / "resolved.yaml")
    elif args.command == "pipeline":
        from .orchestration import Pipeline

        pipeline = Pipeline.from_file(args.config, args.output, args.set)
        if args.dry_run:
            result = pipeline.plan()
        else:
            state = pipeline.run(args.resume)
            result = {
                "name": state["name"],
                "status": "completed",
                "stages": {key: value["status"] for key, value in state["stages"].items()},
                "state": str(pipeline.output / "pipeline.json"),
            }
    elif args.command == "sweep":
        from .orchestration.sweep import Sweep

        sweep = Sweep.from_file(args.config, args.output)
        if args.dry_run:
            result = sweep.plan()
        else:
            state = sweep.run(args.resume)
            result = {
                "status": "completed",
                "runs": len(state["jobs"]),
                "report": str(sweep.output / "summary.md"),
            }
    elif args.command == "report":
        from .orchestration.sweep import summarize_sweep

        result = summarize_sweep(args.sweep)
    elif args.command == "components":
        from .models.backbones import BACKBONES
        from .models.fusion import FUSIONS

        result = {
            "backbones": BACKBONES.names,
            "fusions": FUSIONS.names,
            "tasks": ["classification", "diffusion", "autoencoder"],
            "optimizers": ["sgd", "adam", "adamw"],
            "precision": ["fp32", "bf16", "fp16"],
        }
    elif args.command == "smoke":
        from .demo import smoke

        result = smoke(args.output)
    else:
        parser.error("Unknown command")
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return result
