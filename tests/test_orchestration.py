from dataclasses import asdict
import json
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from pvaug.config import TrainConfig, ModelConfig, write_json
from pvaug.demo import make_demo
from pvaug.orchestration import Pipeline, Stage
from pvaug.orchestration.sweep import Sweep, summarize_sweep

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_artifact_references_and_dry_plan_do_not_touch_data(tmp_path):
    pipeline = Pipeline.from_file(ROOT / "configs/pipelines/paper.yaml", tmp_path / "run")
    plan = pipeline.plan()
    assert [stage["id"] for stage in plan["stages"]] == [
        "estimator",
        "generator",
        "synthetic",
        "recognizer",
        "evaluation",
    ]
    student = plan["stages"][3]
    assert student["config"]["teacher"] == str(tmp_path / "run/estimator/best.pt")
    assert student["depends_on"] == ["estimator", "synthetic"]
    assert not (tmp_path / "run").exists()


def test_pipeline_cycles_unknown_references_and_path_ids_fail(tmp_path):
    with pytest.raises(ValueError, match="cycle"):
        Pipeline(
            "bad",
            tmp_path,
            [Stage("a", "classification", "x", ["b"]), Stage("b", "classification", "x", ["a"])],
        )
    with pytest.raises(ValueError, match="Unknown artifact"):
        Pipeline(
            "bad",
            tmp_path,
            [
                Stage("a", "classification", "x"),
                Stage("b", "classification", "x", inputs={"teacher": "@a.unknown"}),
            ],
        )
    with pytest.raises(ValueError, match="path separators"):
        Stage("../unsafe", "classification", "x")


def test_pipeline_executes_reuses_and_detects_changed_artifacts(tmp_path):
    manifest = make_demo(tmp_path / "data")
    config = TrainConfig(
        manifest=str(manifest),
        image_size=32,
        batch_size=2,
        epochs=1,
        max_steps=1,
        device="cpu",
        model=ModelConfig(fusion=False),
    )
    write_json(tmp_path / "classifier.json", asdict(config))
    pipeline = Pipeline(
        "small",
        tmp_path / "run",
        [
            Stage("train", "classification", "classifier.json"),
            Stage(
                "test",
                "evaluation",
                inputs={"checkpoint": "@train.checkpoint", "manifest": str(manifest)},
                options={"device": "cpu", "batch_size": 2},
            ),
        ],
        tmp_path,
    )
    state = pipeline.run()
    assert all(stage["status"] == "completed" for stage in state["stages"].values())
    original = (tmp_path / "run/train/best.pt").stat().st_mtime_ns
    pipeline.run(resume=True)
    assert (tmp_path / "run/train/best.pt").stat().st_mtime_ns == original
    (tmp_path / "run/test/metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifacts or inputs changed"):
        pipeline.run(resume=True)


def test_pipeline_records_failure_and_resumes_pending_stage(tmp_path, monkeypatch):
    manifest = make_demo(tmp_path / "data")
    config = TrainConfig(
        manifest=str(manifest),
        image_size=32,
        batch_size=2,
        epochs=1,
        max_steps=1,
        device="cpu",
        model=ModelConfig(fusion=False),
    )
    write_json(tmp_path / "classifier.json", asdict(config))
    pipeline = Pipeline(
        "small", tmp_path / "run", [Stage("train", "classification", "classifier.json")], tmp_path
    )
    original = pipeline.execute_stage

    def fail(stage, resume=False):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(pipeline, "execute_stage", fail)
    with pytest.raises(RuntimeError, match="controlled"):
        pipeline.run()
    state = json.loads((tmp_path / "run/pipeline.json").read_text(encoding="utf-8"))
    assert state["stages"]["train"]["status"] == "failed"
    monkeypatch.setattr(pipeline, "execute_stage", original)
    assert pipeline.run(resume=True)["stages"]["train"]["status"] == "completed"


def test_sweep_plan_and_multiseed_statistics(tmp_path):
    sweep = Sweep.from_file(ROOT / "configs/sweeps/fusion.yaml", tmp_path / "sweep")
    jobs = sweep.plan()
    assert len(jobs) == len({job["id"] for job in jobs}) == 9
    state = {
        "jobs": {
            str(i): {
                "status": "completed",
                "parameters": {"model.fusion_kind": "gated", "experiment.seed": seed},
                "metrics": {"accuracy": accuracy, "f1_macro": accuracy, "f1_weighted": accuracy},
            }
            for i, (seed, accuracy) in enumerate([(42, 0.6), (43, 0.8)])
        }
    }
    write_json(tmp_path / "sweep.json", state)
    result = summarize_sweep(tmp_path / "sweep.json")
    assert result[0]["runs"] == 2 and result[0]["accuracy_mean"] == pytest.approx(0.7)
    assert result[0]["accuracy_std"] == pytest.approx(2**0.5 * 0.1)
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "summary.md").exists()


def test_sweep_runs_two_seeds_and_reuses_completed_runs(tmp_path):
    manifest = make_demo(tmp_path / "data")
    doc = OmegaConf.load(ROOT / "configs/base/classification.yaml")
    doc.data.manifest = str(manifest)
    doc.data.image_size = 32
    doc.model.fusion = False
    doc.trainer.epochs, doc.trainer.batch_size, doc.trainer.max_steps = 1, 2, 1
    doc.trainer.device = "cpu"
    OmegaConf.save(doc, tmp_path / "base.yaml")
    sweep = Sweep("base.yaml", {"experiment.seed": [5, 6]}, tmp_path / "sweep", tmp_path)
    state = sweep.run()
    assert len(state["jobs"]) == 2
    assert all(entry["status"] == "completed" for entry in state["jobs"].values())
    assert len(summarize_sweep(tmp_path / "sweep/sweep.json")) == 1
    sweep.run(resume=True)
