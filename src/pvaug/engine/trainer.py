"""One lifecycle for classification, semantic learning, diffusion, and VQ training."""

from contextlib import nullcontext
from pathlib import Path
import platform
import time
import math
import torch
from .contracts import TrainState, Task
from .callbacks import HistoryWriter, CheckpointWriter, ConsoleLogger
from .optim import build_optimizer, build_scheduler
from ..config import TrainConfig, LDMConfig, config_dict, write_json
from ..utils import atomic_save, load_checkpoint, rng_state, restore_rng


def move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    return value


class Trainer:
    """Single-device training with epoch-boundary resume and pluggable lifecycle callbacks.

    max_steps counts optimizer updates. Accumulation uses sample-weighted gradients,
    including incomplete final groups. Checkpoints are written with no pending gradients.
    """

    def __init__(
        self,
        task: Task,
        config: TrainConfig | LDMConfig,
        loader_factory,
        device,
        audit=None,
        metadata=None,
        callbacks=(),
        save=atomic_save,
    ):
        self.task, self.config = task, config
        self.loader_factory, self.device = loader_factory, torch.device(device)
        self.audit, self.metadata = audit, metadata or {}
        self.optimizer = build_optimizer(task.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        precision = config.engine.precision
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires CUDA; CPU supports fp32 or bf16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.state = TrainState()
        self.resumed = False
        self.save = save
        self.callbacks = [HistoryWriter(), *callbacks, CheckpointWriter(save), ConsoleLogger()]

    def emit(self, event, *args):
        for callback in self.callbacks:
            getattr(callback, event)(self, *args)

    def autocast(self):
        precision = self.config.engine.precision
        if precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(self.device.type, dtype=dtype)

    def checkpoint(self):
        return {
            "kind": self.task.kind,
            "format_version": 2,
            "model": self.task.model.state_dict(),
            "config": config_dict(self.config),
            "epoch": self.state.epoch,
            "step": self.state.step,
            "best": self.state.best,
            "skipped_updates": self.state.skipped_updates,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "rng": rng_state(),
            "audit": self.audit,
            **self.task.state_dict(),
            **self.metadata,
        }

    def resume(self, path):
        checkpoint = load_checkpoint(path)
        if checkpoint["kind"] != self.task.kind:
            raise ValueError(f"Expected {self.task.kind} checkpoint")
        old = type(self.config)(**checkpoint["config"])
        previous, current = config_dict(old), config_dict(self.config)
        for key in current:
            if key not in {"output", "device", "workers"} and previous[key] != current[key]:
                raise ValueError(f"Resume config differs: {key}")
        if checkpoint["audit"] != self.audit:
            raise ValueError("Resume data changed")
        self.task.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.task.load_state_dict(checkpoint)
        self.state = TrainState(
            epoch=checkpoint["epoch"],
            step=checkpoint["step"],
            best=checkpoint.get("best", -float("inf")),
            skipped_updates=checkpoint.get("skipped_updates", 0),
        )
        restore_rng(checkpoint["rng"])
        self.resumed = True
        output = Path(self.config.output)
        output.mkdir(parents=True, exist_ok=True)
        if self.task.artifact == "best.pt" and not (output / "best.pt").exists():
            previous_best = Path(path).parent / "best.pt"
            if not previous_best.exists():
                raise ValueError("Resuming into a new output requires the previous best.pt")
            best = load_checkpoint(previous_best)
            if best["epoch"] > checkpoint["epoch"]:
                raise ValueError("best.pt is newer than the resume checkpoint")
            self.save(best, output / "best.pt")
        if not (output / "last.pt").exists():
            self.save(self.checkpoint(), output / "last.pt")

    def reached_limit(self):
        return self.config.max_steps is not None and self.state.step >= self.config.max_steps

    def fit(self, resume=None):
        output = Path(self.config.output)
        if (output / "last.pt").exists() and resume is None:
            raise FileExistsError(f"{output}/last.pt exists; use --resume or a new output")
        output.mkdir(parents=True, exist_ok=True)
        if resume:
            self.resume(resume)
        write_json(output / "config.json", config_dict(self.config))
        write_json(output / "data_audit.json", self.audit)
        write_json(
            output / "environment.json",
            {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": str(self.device),
                "precision": self.config.engine.precision,
                "cuda": torch.version.cuda,
                "platform": platform.platform(),
            },
        )
        self.emit("on_fit_start")
        try:
            for epoch in range(self.state.epoch + 1, self.config.epochs):
                if self.reached_limit():
                    break
                self.state.epoch = epoch
                begin = time.perf_counter()
                train = self.train_epoch(epoch)
                validation = self.task.validate(self.device)
                score = self.task.score(train, validation)
                if not math.isfinite(score):
                    raise RuntimeError("Non-finite model selection score")
                self.state.improved = score > self.state.best
                self.state.best = max(self.state.best, score)
                if self.scheduler:
                    self.scheduler.step()
                self.state.record = {
                    "epoch": epoch + 1,
                    "step": self.state.step,
                    "skipped_updates": self.state.skipped_updates,
                    "train": train,
                    "val": validation,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "seconds": time.perf_counter() - begin,
                    **self.task.epoch_metadata(epoch),
                }
                self.emit("on_epoch_end")
            self.emit("on_fit_end")
        except Exception as error:
            self.emit("on_exception", error)
            raise
        artifact = output / self.task.artifact
        if not artifact.exists():
            raise RuntimeError("Training produced no artifact")
        return artifact

    def train_epoch(self, epoch):
        self.task.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loader = self.loader_factory(epoch)
        totals, seen, group_seen = {}, 0, 0
        if not len(loader):
            raise ValueError("Empty training loader")
        for index, batch in enumerate(loader):
            batch = move_batch(batch, self.device)
            with self.autocast():
                result = self.task.train_step(batch, epoch)
            if result.loss.ndim != 0 or not torch.isfinite(result.loss) or result.size < 1:
                raise RuntimeError("Task returned a non-finite or non-scalar loss")
            # Sum sample losses, then divide gradients once by actual group size.
            self.scaler.scale(result.loss * result.size).backward()
            seen += result.size
            group_seen += result.size
            for key, value in {"loss": result.loss.detach(), **result.metrics}.items():
                totals[key] = totals.get(key, 0.0) + float(value) * result.size
            boundary = (index + 1) % self.config.engine.accumulation_steps == 0
            if boundary or index + 1 == len(loader):
                self.scaler.unscale_(self.optimizer)
                parameters = [p for p in self.task.model.parameters() if p.grad is not None]
                for parameter in parameters:
                    parameter.grad.div_(group_seen)
                finite = all(torch.isfinite(parameter.grad).all() for parameter in parameters)
                if not finite:
                    if not self.scaler.is_enabled():
                        raise RuntimeError("Non-finite gradients")
                    # GradScaler skips an overflowed fp16 update and lowers its scale.
                    # The logical step and EMA advance only for actual optimizer updates.
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    group_seen = 0
                    self.state.skipped_updates += 1
                    continue
                if self.config.engine.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        parameters, self.config.engine.clip_grad_norm, error_if_nonfinite=True
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                group_seen = 0
                self.state.step += 1
                self.task.after_optimizer_step()
                self.emit("on_optimizer_step")
                if self.reached_limit():
                    break
        return {key: value / seen for key, value in totals.items()}
