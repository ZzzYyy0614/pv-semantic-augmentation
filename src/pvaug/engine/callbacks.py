"""Artifact production is isolated from optimization and method code."""

import json
from pathlib import Path
from .contracts import Callback


class HistoryWriter(Callback):
    def on_fit_start(self, trainer):
        self.path = Path(trainer.config.output) / "history.jsonl"
        if not trainer.resumed:
            self.path.write_text("", encoding="utf-8")

    def on_epoch_end(self, trainer):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trainer.state.record) + "\n")


class CheckpointWriter(Callback):
    def __init__(self, save):
        self.save = save

    def on_epoch_end(self, trainer):
        state = trainer.state
        output = Path(trainer.config.output)
        checkpoint = trainer.checkpoint()
        # last.pt is always current. Numbered snapshots are independently configurable.
        self.save(checkpoint, output / "last.pt")
        if state.improved:
            self.save(checkpoint, output / "best.pt")
        if trainer.task.artifact not in {"last.pt", "best.pt"}:
            self.save(checkpoint, output / trainer.task.artifact)
        interval = trainer.config.engine.checkpoint_every
        if interval > 1 and (state.epoch + 1) % interval == 0:
            self.save(checkpoint, output / "checkpoints" / f"epoch_{state.epoch + 1:04d}.pt")


class ConsoleLogger(Callback):
    def on_epoch_end(self, trainer):
        print(json.dumps(trainer.state.record), flush=True)
