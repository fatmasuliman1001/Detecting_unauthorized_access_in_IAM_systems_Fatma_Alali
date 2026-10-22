"""
evaluate.py
===========
Metrics and plots for Chapter 4 (Section 3.7 evaluation approach):
accuracy, precision, recall, F1, AUC-ROC, confusion matrix, ROC curve,
and SHAP-based feature importance for the Random Forest component.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)


def compute_metrics(y_true, y_pred, y_proba=None, model_name="model"):
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["auc_roc"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else 0.0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else 0.0
    return metrics


def results_table(list_of_metric_dicts):
    return pd.DataFrame(list_of_metric_dicts).set_index("model").round(4)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax = ax or plt.gca()
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Threat"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Threat"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_roc(y_true, y_proba, label="model", ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax = ax or plt.gca()
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    return ax


def shap_feature_importance(rf_model, X, feature_names, max_display=10):
    """
    Returns a SHAP-based feature importance DataFrame for the Random
    Forest component (Section 3.6). Requires the `shap` package.
    """
    import shap
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        # older shap: list of arrays, one per class
        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        vals = np.asarray(shap_values)
        if vals.ndim == 3:
            # newer shap: (n_samples, n_features, n_classes)
            class_idx = 1 if vals.shape[-1] > 1 else 0
            vals = vals[:, :, class_idx]

    mean_abs = np.abs(vals).mean(axis=0).ravel()
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp.head(max_display)


def cross_dataset_summary(same_domain_metrics, cross_domain_metrics):
    """
    Builds the RQ3 comparison table: performance when trained/tested on the
    same dataset vs. trained on one dataset and tested on the other.
    """
    rows = []
    for name, m in [("Same-dataset", same_domain_metrics), ("Cross-dataset", cross_domain_metrics)]:
        row = dict(m)
        row["setting"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("setting").round(4)
