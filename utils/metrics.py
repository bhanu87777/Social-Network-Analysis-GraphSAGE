# utils/metrics.py
import torch
from sklearn.metrics import accuracy_score, classification_report

def evaluate_logits(logits, labels, mask):
    preds = logits.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = mask.cpu().numpy()
    acc = accuracy_score(labels[mask], preds[mask])
    report = classification_report(labels[mask], preds[mask])
    return acc, report
