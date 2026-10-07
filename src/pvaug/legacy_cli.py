import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from .config import CLASSES, TrainConfig, LDMConfig, read_config, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EL defect recognition with semantic-aware augmentation"
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU Torch threads (default 4)")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Import class folders / preserve existing test split"
    )
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", default="data/real.csv")
    prepare.add_argument("--classes", help="JSON list in original checkpoint class order")
    prepare.add_argument("--groups", help="CSV image,group for cells belonging to a module")
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)
    audit = commands.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--classes")
    audit.add_argument("--output")
    legacy = commands.add_parser("import-legacy", help="Import image;mask;integer-label EL lists")
    legacy.add_argument("--train", required=True)
    legacy.add_argument("--val")
    legacy.add_argument("--test")
    legacy.add_argument("--classes", required=True, help="JSON list in original integer-ID order")
    legacy.add_argument("--path-map", help="JSON old server path prefix -> local path prefix")
    legacy.add_argument("--output", default="data/real.csv")
    demo = commands.add_parser("demo-data")
    demo.add_argument("--output", default="data/demo")
    smoke = commands.add_parser("smoke", help="Run the complete pipeline on toy images on CPU")
    smoke.add_argument("--output", default="runs/smoke")
    for name in ["train", "train-generator", "train-autoencoder"]:
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        for field in ["manifest", "output", "device", "teacher", "generated-manifest"]:
            if field in {"teacher", "generated-manifest"} and name != "train":
                continue
            sub.add_argument("--" + field)
        sub.add_argument("--epochs", type=int)
        sub.add_argument("--max-steps", type=int)
        if name != "train-autoencoder":
            sub.add_argument("--resume")
        if name == "train":
            sub.add_argument("--init", help="Raw recovered .pth or pvaug checkpoint")
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", default="runs/eval")
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--batch-size", type=int, default=24)
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", default="runs/predict")
    predict.add_argument("--device", default="auto")
    predict.add_argument("--gradcam", action="store_true")
    generate = commands.add_parser("generate")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--sketches", required=True, help="CSV mask,label")
    generate.add_argument("--output", default="data/generated")
    generate.add_argument("--per-class", type=int, default=200)
    generate.add_argument("--steps", type=int, default=50)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--eta", type=float, default=0.0)
    generate.add_argument("--device", default="auto")
    convert = commands.add_parser("convert-generator")
    convert.add_argument(
        "--config", required=True, help="Explicit original class order and backup architecture"
    )
    convert.add_argument("--checkpoint", required=True)
    convert.add_argument("--output", required=True)
    convert_classifier = commands.add_parser("convert-classifier")
    convert_classifier.add_argument(
        "--config", required=True, help="Known architecture and class order"
    )
    convert_classifier.add_argument("--checkpoint", required=True)
    convert_classifier.add_argument("--output", required=True)
    convert_classifier.add_argument(
        "--training-manifest", help="Declare verified original real split"
    )
    bench = commands.add_parser("benchmark")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--device", default="auto")
    bench.add_argument("--iterations", type=int, default=100)
    bench.add_argument("--output", default="runs/benchmark.json")
    hist = commands.add_parser("confidence")
    hist.add_argument("--predictions", required=True)
    hist.add_argument("--output", default="runs/confidence.json")
    hist.add_argument("--bins", type=int, default=10)
    exp = commands.add_parser("experiments")
    exp.add_argument("--config", required=True)
    exp.add_argument(
        "--suite", choices=["components", "temperature", "ratio", "backbones"], default="components"
    )
    exp.add_argument("--seeds", default="42")
    exp.add_argument("--output", default="runs/experiments")
    exp.add_argument(
        "--run", action="store_true", help="Train and evaluate; default writes reviewable configs"
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    result = None
    if args.command in {"prepare", "audit"}:
        from .data import prepare_folders, read_manifest, audit_samples

        classes = (
            json.loads(Path(args.classes).read_text(encoding="utf-8")) if args.classes else CLASSES
        )
        if args.command == "prepare":
            result = prepare_folders(
                args.source,
                args.output,
                classes,
                args.val_fraction,
                args.test_fraction,
                args.seed,
                args.groups,
            )
        else:
            result = audit_samples(read_manifest(args.manifest, classes))
            if args.output:
                write_json(args.output, result)
    elif args.command == "import-legacy":
        from .data import import_legacy_lists

        classes = json.loads(Path(args.classes).read_text(encoding="utf-8"))
        mapping = (
            json.loads(Path(args.path_map).read_text(encoding="utf-8")) if args.path_map else None
        )
        result = import_legacy_lists(args.output, classes, args.train, args.val, args.test, mapping)
    elif args.command in {"demo-data", "smoke"}:
        from .demo import make_demo, smoke

        result = (smoke if args.command == "smoke" else make_demo)(args.output)
    elif args.command.startswith("train"):
        kind = TrainConfig if args.command == "train" else LDMConfig
        config = read_config(args.config, kind)
        overrides = {
            name: getattr(args, name)
            for name in [
                "manifest",
                "output",
                "device",
                "epochs",
                "max_steps",
                "teacher",
                "generated_manifest",
            ]
            if getattr(args, name, None) is not None
        }
        config = replace(config, **overrides)
        if args.command == "train":
            from .train import train_classifier

            result = train_classifier(config, args.resume, args.init)
        else:
            from .generation import train_generator, train_autoencoder

            result = (
                train_generator(config, args.resume)
                if args.command == "train-generator"
                else train_autoencoder(config)
            )
    elif args.command == "eval":
        from .train import evaluate_checkpoint

        result = evaluate_checkpoint(
            args.checkpoint, args.manifest, args.output, args.split, args.device, args.batch_size
        )
    elif args.command == "predict":
        from .inference import predict

        result = predict(args.checkpoint, args.input, args.output, args.device, args.gradcam)
    elif args.command == "generate":
        from .generation import generate_samples

        result = generate_samples(
            args.checkpoint,
            args.sketches,
            args.output,
            args.per_class,
            args.steps,
            args.seed,
            args.device,
            args.eta,
        )
    elif args.command == "convert-generator":
        from .generation import convert_legacy_generator

        result = convert_legacy_generator(
            read_config(args.config, LDMConfig), args.checkpoint, args.output
        )
    elif args.command == "convert-classifier":
        from .train import convert_legacy_classifier

        result = convert_legacy_classifier(
            read_config(args.config), args.checkpoint, args.output, args.training_manifest
        )
    elif args.command == "benchmark":
        from .inference import benchmark

        result = benchmark(args.checkpoint, args.device, iterations=args.iterations)
        write_json(args.output, result)
    elif args.command == "confidence":
        from .inference import confidence_histogram

        result = confidence_histogram(args.predictions, args.output, args.bins)
    elif args.command == "experiments":
        from .experiments import experiment_plan

        config = read_config(args.config)
        paths = experiment_plan(
            config, args.output, args.suite, tuple(map(int, args.seeds.split(",")))
        )
        if args.run:
            from .train import train_classifier, evaluate_checkpoint

            results = {}
            for path in paths:
                experiment = read_config(path)
                checkpoint = train_classifier(experiment)
                results[path.stem] = evaluate_checkpoint(
                    checkpoint, experiment.manifest, Path(experiment.output) / "test"
                )
            write_json(Path(args.output) / "results.json", results)
            result = results
        else:
            result = [str(p) for p in paths]
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
