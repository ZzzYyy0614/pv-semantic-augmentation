import pytest
import torch
from pvaug.models import DefectClassifier
from pvaug.config import ModelConfig
from pvaug.registry import Registry


@pytest.mark.parametrize("fusion", ["additive", "concat"])
def test_registered_fusion_is_used_by_classifier_and_gets_gradients(fusion):
    model = DefectClassifier(ModelConfig(variant="paper", fusion_kind=fusion))
    image = torch.rand(2, 3, 32, 32)
    logits = model(image)
    logits.square().mean().backward()
    assert logits.shape == (2, 5)
    assert any(parameter.grad is not None for parameter in model.fusions[0].parameters())


def test_registry_duplicate_and_unknown_component_fail():
    registry = Registry("test")
    registry.register("identity")(lambda value: value)
    assert registry.build("identity", value=3) == 3
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register("identity")(lambda value: value)
    with pytest.raises(ValueError, match="available: identity"):
        registry.build("missing")


@pytest.mark.parametrize("backbone", ["convnext_tiny", "swin_t"])
def test_backbone_contract_works_under_a_new_registered_name(backbone, monkeypatch):
    from pvaug.models.backbones import BACKBONES

    monkeypatch.setattr(BACKBONES, "_factories", BACKBONES._factories.copy())
    name = "contract_alias"

    @BACKBONES.register(name)
    def build_alias(pretrained=False):
        return BACKBONES.build(backbone, pretrained=pretrained)

    original = DefectClassifier(ModelConfig(variant="paper", backbone=backbone)).eval()
    alias = DefectClassifier(ModelConfig(variant="paper", backbone=name)).eval()
    alias.load_state_dict(original.state_dict(), strict=True)
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(original(image), alias(image), atol=0, rtol=0)
