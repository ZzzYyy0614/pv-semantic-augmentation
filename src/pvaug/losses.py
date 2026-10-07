import torch.nn.functional as F


def linear_alpha(epoch, epochs, start=0.9, end=0.1):
    """Epoch 0 uses start; the final epoch uses end (unreported paper endpoints)."""
    if epochs < 1 or not 0 <= epoch < epochs:
        raise ValueError("Invalid epoch")
    return start if epochs == 1 else start + (end - start) * epoch / (epochs - 1)


def semantic_loss(student_logits, labels, teacher_logits=None, temperature=15.0, alpha=0.5):
    """Eq. (2): KL(teacher || student) on both real and generated samples."""
    ce = F.cross_entropy(student_logits, labels)
    if teacher_logits is None:
        return ce, {"ce": ce.detach(), "kd": ce.detach() * 0}
    if temperature <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Invalid temperature or alpha")
    # Temperature-scaled KL stays in fp32 under autocast to avoid cancellation.
    teacher_prob = F.softmax(teacher_logits.detach().float() / temperature, dim=1)
    student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=1)
    kd = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature**2
    return (1 - alpha) * ce + alpha * kd, {"ce": ce.detach(), "kd": kd.detach()}
