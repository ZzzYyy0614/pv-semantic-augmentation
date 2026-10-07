import numpy as np


def classification_metrics(targets, predictions, classes):
    n = len(classes)
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("Predictions and nonempty targets must have the same shape")
    if (
        np.any(targets < 0)
        or np.any(targets >= n)
        or np.any(predictions < 0)
        or np.any(predictions >= n)
    ):
        raise ValueError("Class index out of range")
    matrix = np.bincount(n * targets + predictions, minlength=n * n).reshape(n, n)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(n), where=precision + recall != 0
    )
    return {
        "accuracy": float(tp.sum() / targets.size),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float((f1 * support).sum() / targets.size),
        "samples": int(targets.size),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
    }
