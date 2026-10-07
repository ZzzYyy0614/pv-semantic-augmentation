"""Recovered CompVis U-Net and VQ first stage, without legacy Lightning dependencies."""

from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from ..vendor.ldm.modules.diffusionmodules.model import Encoder, Decoder
from ..vendor.ldm.modules.diffusionmodules.openaimodel import UNetModel
from ..vendor.quantize import VectorQuantizer2
from ..utils import load_checkpoint


class VQFirstStage(nn.Module):
    """Same parameter names and pre-quant encode / quantized decode as VQModelInterface."""

    def __init__(self, config):
        super().__init__()
        ddconfig = dict(
            double_z=False,
            z_channels=config.latent_channels,
            resolution=config.image_size,
            in_channels=3,
            out_ch=3,
            ch=config.first_stage_channels,
            ch_mult=config.first_stage_mult,
            num_res_blocks=config.first_stage_res_blocks,
            attn_resolutions=[],
            dropout=0.0,
        )
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer2(config.n_embed, config.latent_channels, beta=0.25)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, 1)

    def encode(self, image):
        return self.quant_conv(self.encoder(image))

    def decode(self, latent):
        quantized, _, _ = self.quantize(latent)
        return self.decoder(self.post_quant_conv(quantized))

    def reconstruction_loss(self, images):
        quantized, codebook_loss, _ = self.quantize(self.encode(images))
        reconstruction = self.decoder(self.post_quant_conv(quantized))
        return F.l1_loss(reconstruction, images) + codebook_loss

    def load_pretrained(self, path):
        raw = load_checkpoint(path)
        state = raw.get("state_dict", raw.get("model", raw))
        if any(k.startswith("first_stage_model.") for k in state):
            state = {
                k.removeprefix("first_stage_model."): v
                for k, v in state.items()
                if k.startswith("first_stage_model.")
            }
        expected = self.state_dict()
        # Discriminator and perceptual-loss weights from the original AE are not inference layers.
        state = {k: v for k, v in state.items() if k in expected}
        self.load_state_dict(state, strict=True)


class SketchLDM(nn.Module):
    def __init__(self, config, load_first_stage=True):
        super().__init__()
        self.config = config
        self.first_stage = VQFirstStage(config).requires_grad_(False).eval()
        if load_first_stage and not config.smoke_random_first_stage:
            if not config.first_stage_checkpoint:
                raise ValueError("A trained VQ first stage is required")
            self.first_stage.load_pretrained(config.first_stage_checkpoint)
        factor = 2 ** (len(config.first_stage_mult) - 1)
        self.unet = UNetModel(
            image_size=config.image_size // factor,
            in_channels=2 * config.latent_channels,
            out_channels=config.latent_channels,
            model_channels=config.model_channels,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            channel_mult=config.channel_mult,
            num_heads=config.num_heads,
            resblock_updown=True,
            use_checkpoint=config.gradient_checkpointing,
            use_spatial_transformer=config.cross_attention,
            context_dim=config.cross_attention_dim if config.cross_attention else None,
        )
        self.class_embedding = nn.Embedding(len(config.classes), config.cross_attention_dim)
        if not config.cross_attention:
            self.class_embedding.requires_grad_(False)
        if config.beta_schedule == "scaled_linear":
            betas = torch.linspace(
                config.beta_start**0.5, config.beta_end**0.5, config.timesteps, dtype=torch.float64
            ).square()
        elif config.beta_schedule == "linear":
            betas = torch.linspace(
                config.beta_start, config.beta_end, config.timesteps, dtype=torch.float64
            )
        else:
            raise ValueError("beta_schedule must be scaled_linear or linear")
        self.register_buffer("alpha_bar", (1 - betas).cumprod(0).float())

    def train(self, mode=True):
        super().train(mode)
        self.first_stage.eval()
        return self

    @torch.no_grad()
    def encode(self, images, sketches):
        latent = self.first_stage.encode(images * 2 - 1) * self.config.scale_factor
        condition = self.first_stage.encode(sketches.repeat(1, 3, 1, 1) * 2 - 1)
        return latent, condition

    def predict_noise(self, noisy, timesteps, condition, labels):
        tokens = self.class_embedding(labels).unsqueeze(1) if self.config.cross_attention else None
        return self.unet(torch.cat([noisy, condition], dim=1), timesteps, context=tokens)

    def loss(self, images, sketches, labels):
        latent, condition = self.encode(images, sketches)
        noise = torch.randn_like(latent)
        timesteps = torch.randint(len(self.alpha_bar), (len(images),), device=images.device)
        alpha = self.alpha_bar[timesteps, None, None, None]
        noisy = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
        prediction = self.predict_noise(noisy, timesteps, condition, labels)
        return (F.mse_loss if self.config.loss_type == "l2" else F.l1_loss)(prediction, noise)

    @torch.no_grad()
    def sample(self, sketches, labels, steps=50, seed=42, eta=0.0):
        if not 1 <= steps <= len(self.alpha_bar) or eta < 0:
            raise ValueError("Invalid sampling steps or eta")
        device = sketches.device
        generator = torch.Generator(device=device).manual_seed(seed)
        condition = self.first_stage.encode(sketches.repeat(1, 3, 1, 1) * 2 - 1)
        latent = torch.randn(condition.shape, device=device, generator=generator)
        times = torch.linspace(len(self.alpha_bar) - 1, 0, steps, device=device).round().long()
        for index, time in enumerate(times):
            t = time.expand(len(sketches))
            epsilon = self.predict_noise(latent, t, condition, labels)
            alpha = self.alpha_bar[time]
            previous = (
                self.alpha_bar[times[index + 1]]
                if index + 1 < len(times)
                else latent.new_tensor(1.0)
            )
            x0 = (latent - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
            sigma = (
                eta * (((1 - previous) / (1 - alpha)) * (1 - alpha / previous)).clamp_min(0).sqrt()
            )
            direction = (1 - previous - sigma.square()).clamp_min(0).sqrt() * epsilon
            latent = previous.sqrt() * x0 + direction
            if eta:
                latent += sigma * torch.randn(latent.shape, device=device, generator=generator)
        return ((self.first_stage.decode(latent / self.config.scale_factor) + 1) / 2).clamp(0, 1)

    def load_legacy(self, path):
        raw = load_checkpoint(path)
        state = raw.get("state_dict", raw)
        target = self.state_dict()
        mapped = {}
        mapping = [
            ("model.diffusion_model.", "unet."),
            ("first_stage_model.", "first_stage."),
            ("cond_stage2_model.embedding.", "class_embedding."),
        ]
        for old, new in mapping:
            mapped.update({new + k[len(old) :]: v for k, v in state.items() if k.startswith(old)})
        mapped["alpha_bar"] = state.get("alphas_cumprod", target["alpha_bar"])
        self.load_state_dict(mapped, strict=True)


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.detach(), 1 - decay)

    def state_dict(self):
        return {"shadow": self.shadow, "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state):
        self.updates, self.decay = state["updates"], state["decay"]
        if self.shadow.keys() != state["shadow"].keys():
            raise ValueError("EMA parameter names differ")
        self.shadow = {
            name: value.to(self.shadow[name].device) for name, value in state["shadow"].items()
        }

    @contextmanager
    def apply(self, model):
        parameters = dict(model.named_parameters())
        backup = {name: parameters[name].detach().clone() for name in self.shadow}
        with torch.no_grad():
            for name, value in self.shadow.items():
                parameters[name].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)
