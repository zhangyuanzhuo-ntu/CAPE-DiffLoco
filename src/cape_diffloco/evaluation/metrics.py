import numpy as np


def success_rate(success_flags):
    success_flags = np.asarray(success_flags).astype(float)
    return float(success_flags.mean()) if success_flags.size else float('nan')


def fall_rate(fall_flags):
    fall_flags = np.asarray(fall_flags).astype(float)
    return float(fall_flags.mean()) if fall_flags.size else float('nan')


def contact_f1(pred, target, threshold=0.5, eps=1e-8):
    pred = (np.asarray(pred) > threshold).astype(int)
    target = np.asarray(target).astype(int)
    tp = ((pred == 1) & (target == 1)).sum()
    fp = ((pred == 1) & (target == 0)).sum()
    fn = ((pred == 0) & (target == 1)).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
