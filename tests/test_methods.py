from dataclasses import replace
import math
import pytest
import torch
from pvaug.config import ModelConfig, LDMConfig
from pvaug.losses import semantic_loss, linear_alpha
from pvaug.metrics import classification_metrics
from pvaug.models import DefectClassifier
from pvaug.models.edge import SobelPrior
from pvaug.models.sketch_ldm import SketchLDM
from pvaug.train import initialize_weights


def test_kd_has_correct_direction_scaling_and_frozen_teacher():
    student = torch.tensor([[0.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[math.log(3), 0.0]], requires_grad=True)
    loss, parts = semantic_loss(student, torch.tensor([0]), teacher, temperature=1.0, alpha=1.0)
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[0, 0] < 0  # Gradient descent increases the teacher-preferred class logit.
    assert torch.isfinite(parts["kd"])


def test_temperature_stability_and_supervised_endpoint():
    torch.manual_seed(0)
    logits, teacher = torch.randn(3, 5, requires_grad=True), torch.randn(3, 5)
    labels = torch.tensor([0, 1, 2])
    ce, _ = semantic_loss(logits, labels)
    loss, _ = semantic_loss(logits, labels, teacher, 15.0, alpha=0.0)
    assert torch.allclose(loss, ce)
    scaled, parts = semantic_loss(logits, labels, teacher, 15.0, alpha=0.7)
    assert torch.isfinite(scaled) and parts["kd"] > 0
    assert linear_alpha(0, 100, 0.7, 0.3) == 0.7
    assert linear_alpha(99, 100, 0.7, 0.3) == pytest.approx(0.3)


def test_macro_metrics_do_not_hide_minority_class_failure():
    result = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0], ["normal", "defect"])
    assert result["accuracy"] == 0.75
    assert result["recall_macro"] == 0.5
    assert result["precision_macro"] == 0.375
    assert result["f1_macro"] == pytest.approx(3 / 7)
    assert result["confusion_matrix"] == [[3, 0], [1, 0]]


def test_sobel_flat_image_and_vertical_step():
    sobel = SobelPrior()
    assert len(list(sobel.parameters())) == 0
    assert torch.count_nonzero(sobel(torch.zeros(2, 3, 16, 16))) == 0
    image = torch.zeros(1, 3, 16, 16)
    image[..., 8:] = 1
    edge = sobel(image)
    assert edge[..., 7:9].mean() > 0.9
    assert edge[..., :6].max() == 0


@pytest.mark.parametrize("backbone", ["resnet18", "convnext_tiny"])
def test_recovered_wrapper_matches_original_forward(backbone):
    torch.manual_seed(2)
    model = DefectClassifier(ModelConfig(backbone=backbone, variant="recovered")).eval()
    image, edge = torch.rand(1, 3, 64, 64), torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        actual = model(image, edge)
        expected = model.core((image - model.mean) / model.std, (edge - 0.485) / 0.229)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_raw_ddp_weights_load_strictly(tmp_path):
    original = DefectClassifier(ModelConfig()).eval()
    path = tmp_path / "legacy.pth"
    torch.save({"module." + k: v for k, v in original.core.state_dict().items()}, path)
    restored = DefectClassifier(ModelConfig()).eval()
    initialize_weights(restored, path)
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    bad = tmp_path / "incomplete.pth"
    torch.save({"fc.weight": original.core.fc.weight}, bad)
    with pytest.raises(RuntimeError, match="Missing key"):
        initialize_weights(restored, bad)


@pytest.mark.parametrize("backbone", ["resnet18", "convnext_tiny", "swin_t"])
def test_paper_backbones_forward_and_backward(backbone):
    model = DefectClassifier(ModelConfig(backbone=backbone, variant="paper")).train()
    prediction = model(torch.rand(2, 3, 64, 64))
    assert prediction.shape == (2, 5)
    prediction.square().mean().backward()
    assert model.fusions[0].gate[0].weight.grad is not None


def tiny_ldm(**overrides):
    config = LDMConfig(
        image_size=32,
        first_stage_checkpoint=None,
        smoke_random_first_stage=True,
        first_stage_channels=32,
        first_stage_mult=[1, 2],
        first_stage_res_blocks=1,
        model_channels=32,
        channel_mult=[1, 2],
        num_res_blocks=1,
        num_heads=4,
        attention_resolutions=[1, 2],
        cross_attention_dim=32,
        timesteps=10,
    )
    return replace(config, **overrides)


def test_ldm_class_control_freezes_first_stage_and_sampling_is_repeatable():
    torch.manual_seed(0)
    model = SketchLDM(tiny_ldm(gradient_checkpointing=True))
    # CompVis zero-initializes both denoiser and transformer projections. Unzero both
    # to probe conditioning before training rather than testing the zero initialization.
    torch.nn.init.normal_(model.unet.out[-1].weight, std=0.01)
    for module in model.unet.modules():
        if module.__class__.__name__ == "SpatialTransformer":
            torch.nn.init.normal_(module.proj_out.weight, std=0.01)
    image, sketch = torch.rand(2, 3, 32, 32), torch.rand(2, 1, 32, 32)
    latent, condition = model.encode(image, sketch)
    noise, time = torch.randn_like(latent), torch.tensor([3, 3])
    class0 = model.predict_noise(noise, time, condition, torch.tensor([0, 0]))
    class1 = model.predict_noise(noise, time, condition, torch.tensor([1, 1]))
    assert not torch.allclose(class0, class1)
    loss = model.loss(image, sketch, torch.tensor([0, 1]))
    loss.backward()
    assert model.class_embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.first_stage.parameters())
    model.eval()
    first = model.sample(sketch[:1], torch.tensor([0]), steps=2, seed=123)
    second = model.sample(sketch[:1], torch.tensor([0]), steps=2, seed=123)
    assert torch.equal(first, second)
    assert first.shape == (1, 3, 32, 32)
    assert 0 <= first.min() <= first.max() <= 1


def test_backup_ldm_ignores_classes_and_legacy_conversion_is_exact(tmp_path):
    model = SketchLDM(tiny_ldm(cross_attention=False)).eval()
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("unet."):
            state["model.diffusion_model." + key[5:]] = value
        elif key.startswith("first_stage."):
            state["first_stage_model." + key[12:]] = value
        elif key.startswith("class_embedding."):
            state["cond_stage2_model.embedding." + key[16:]] = value
        else:
            state["alphas_cumprod"] = value
    path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": state}, path)
    restored = SketchLDM(tiny_ldm(cross_attention=False), load_first_stage=False)
    restored.load_legacy(path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    x, condition = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    t = torch.tensor([4])
    assert torch.equal(
        model.predict_noise(x, t, condition, torch.tensor([0])),
        model.predict_noise(x, t, condition, torch.tensor([1])),
    )
