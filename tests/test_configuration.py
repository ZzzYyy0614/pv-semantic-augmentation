from pathlib import Path
import pytest
from omegaconf import OmegaConf
from pvaug.configuration import load_config, resolve_config
from pvaug.config import LDMConfig

ROOT = Path(__file__).resolve().parents[1]


def test_hierarchical_defaults_and_typed_overrides():
    config = load_config(
        ROOT / "configs/tasks/recognizer.yaml",
        overrides=[
            "model.backbone=swin_t",
            "trainer.engine.accumulation_steps=3",
            "trainer.engine.precision=bf16",
            "semantic.teacher=null",
        ],
    )
    assert config.model.variant == "paper" and config.model.backbone == "swin_t"
    assert config.teacher is None and config.engine.accumulation_steps == 3
    assert config.engine.precision == "bf16" and config.lr == 0.0005


def test_composition_resolves_parent_interpolations_after_child_override(tmp_path):
    (tmp_path / "parent.yaml").write_text("name: base\noutput: runs/${name}\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("defaults: [parent.yaml]\nname: study\n", encoding="utf-8")
    assert resolve_config(tmp_path / "child.yaml")["output"] == "runs/study"


@pytest.mark.parametrize("override", ["trainer.epoch=2", "model.backbone", "made_up.key=true"])
def test_override_typo_is_rejected(override):
    with pytest.raises(ValueError):
        load_config(ROOT / "configs/tasks/estimator.yaml", overrides=[override])


def test_cycles_and_unknown_schema_fields_fail(tmp_path):
    (tmp_path / "a.yaml").write_text("defaults: [b.yaml]\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("defaults: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        resolve_config(tmp_path / "a.yaml")
    doc = resolve_config(ROOT / "configs/base/classification.yaml")
    doc["optimizer"]["learning_rate"] = 0.001
    OmegaConf.save(OmegaConf.create(doc), tmp_path / "invalid.yaml")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        load_config(tmp_path / "invalid.yaml")


def test_task_type_and_engine_settings_are_validated():
    with pytest.raises(ValueError, match="cannot be used"):
        load_config(ROOT / "configs/tasks/estimator.yaml", kind=LDMConfig)
    with pytest.raises(ValueError, match="Accumulation"):
        load_config(
            ROOT / "configs/tasks/estimator.yaml", overrides=["trainer.engine.accumulation_steps=0"]
        )
    config = load_config(ROOT / "configs/compatibility/generator.yaml")
    assert not config.cross_attention and config.loss_type == "l1"


def test_typed_schema_rejects_wrong_nested_value_types():
    with pytest.raises(ValueError, match="Invalid typed"):
        load_config(ROOT / "configs/tasks/estimator.yaml", overrides=["trainer.batch_size=abc"])
    with pytest.raises(ValueError, match="Unknown registered backbone"):
        load_config(ROOT / "configs/tasks/estimator.yaml", overrides=["model.backbone=unknown"])
