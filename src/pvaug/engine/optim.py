"""Optimizer construction lives outside task definitions."""

import torch


def build_optimizer(model, config):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The task has no trainable parameters")
    common = dict(lr=config.lr, weight_decay=config.weight_decay)
    constructors = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if config.optimizer == "sgd":
        return torch.optim.SGD(parameters, momentum=getattr(config, "momentum", 0.9), **common)
    return constructors[config.optimizer](parameters, **common)


def build_scheduler(optimizer, config):
    if config.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    return None
