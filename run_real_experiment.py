"""
run_real_experiment.py
=======================
Runs a REAL experiment (no fabricated numbers) using IAM-SynthSim, the
custom agent-based synthetic IAM log simulator built for this study
(Section 3.4). LANL Cyber1 is NOT included here because this sandboxed
environment has no network access to csr.lanl.gov; results below are
therefore reported honestly as being derived from the synthetic simulator
only, and RQ3 is answered using a domain-shift proxy (two differently
parameterized simulation regimes, A and B) rather than a true second
real-world dataset.

DATA SPLITS (important):
  * MAIN EXPERIMENT (Tables 4.2 / 4.3, Figures 4.1-4.3): one stratified
    70/30 train/test split, random_state=42 (21,000 train / 9,000 test
    events, 90 threats in test). There is NO validation subset: model
    hyperparameters and the ensemble weights (0.4 / 0.3 / 0.3) are fixed
    a priori from the approved methodology and are NOT tuned on any data.
  * RQ3 cross-regime analysis (Section 4.4) only: calls
    preprocessing.stratified_split(), a separate 70/15/15 helper.
  * The LSTM's Keras validation_split=0.1 only logs validation loss per
    epoch; it is not used for early stopping or model selection.
"""
import sys, os, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import data_loader as dl
import feature_engineering as fe
import preprocessing as pp
import models as mdl
import evaluate as ev

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
np.random.seed(42)
import random
random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)
except Exception as e:
    print("tf seeding warning:", e)

N_EVENTS = 30_000
SEQ_LEN = 24
LSTM_EPOCHS = 4

log = {}

# ---------------------------------------------------------------------
# 1. Load + feature-engineer Regime A (primary) and Regime B (domain-shift)
# ---------------------------------------------------------------------
t0 = time.time()
raw_a = dl.load_iam_synthsim(n_events=N_EVENTS, regime="A")
feat_a = fe.build_common_features_synthsim(raw_a).reset_index(drop=True)
raw_b = dl.load_iam_synthsim(n_events=N_EVENTS, regime="B")
feat_b = fe.build_common_features_synthsim(raw_b).reset_index(drop=True)

# Save the exact simulated datasets used in this run, so they can be
# inspected or reused without re-running the simulator. Saving does not
# consume any randomness, so it does not affect any results.
os.makedirs("data", exist_ok=True)
raw_a.to_csv("data/iam_synthsim_regime_A.csv", index=False)
raw_b.to_csv("data/iam_synthsim_regime_B.csv", index=False)
print(f"[1] data loaded+featurized in {time.time()-t0:.1f}s")

balance = pd.DataFrame([
    {"dataset": "IAM-SynthSim Regime A", "n_events": len(feat_a),
     "n_threat_events": int(feat_a['label'].sum()),
     "threat_ratio": round(feat_a['label'].mean(), 5)},
    {"dataset": "IAM-SynthSim Regime B", "n_events": len(feat_b),
     "n_threat_events": int(feat_b['label'].sum()),
     "threat_ratio": round(feat_b['label'].mean(), 5)},
]).set_index("dataset")
balance.to_csv("results/4_1_class_balance.csv")
print(balance)

# ---------------------------------------------------------------------
# 2. Build ALIGNED static + sequence arrays for Regime A (per-user order)
# ---------------------------------------------------------------------
t0 = time.time()
aligned_static, aligned_labels = [], []
for _, g in feat_a.groupby("user_id"):
    aligned_static.append(g[fe.STATIC_FEATURES].to_numpy(dtype=float))
X_static_aligned = np.vstack(aligned_static)

X_seq_all, y_seq_all = fe.create_sequences(feat_a, seq_len=SEQ_LEN, stride=1)
assert len(X_seq_all) == len(X_static_aligned)
print(f"[2] aligned arrays built in {time.time()-t0:.1f}s -> {X_static_aligned.shape}, {X_seq_all.shape}")

idx_train, idx_test = tts(np.arange(len(y_seq_all)), test_size=0.3,
                           stratify=y_seq_all, random_state=42)

Xs_tr, Xs_te = X_static_aligned[idx_train], X_static_aligned[idx_test]
Xq_tr, Xq_te = X_seq_all[idx_train], X_seq_all[idx_test]
y_tr, y_te = y_seq_all[idx_train], y_seq_all[idx_test]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(Xs_tr)
Xs_tr_scaled = scaler.transform(Xs_tr)
Xs_te_scaled = scaler.transform(Xs_te)
Xs_tr_res, y_tr_res = pp.resample_training_set(Xs_tr_scaled, y_tr)
print(f"    train (resampled): {Xs_tr_res.shape}, positives={y_tr_res.sum()}")
print(f"    test (real ratio): {Xs_te_scaled.shape}, positives={y_te.sum()}")

# ---------------------------------------------------------------------
# 3. Rule-based baseline
# ---------------------------------------------------------------------
def rule_based_baseline(X_raw, feature_names):
    d = pd.DataFrame(X_raw, columns=feature_names)
    flagged = (
        (d["login_velocity"] > d["login_velocity"].quantile(0.99)) |
        (d["failure_rate"] > 0.5) |
        (d["privilege_change"] == 1)
    )
    return flagged.astype(int).values

y_pred_rule = rule_based_baseline(Xs_te, fe.STATIC_FEATURES)
m_rule = ev.compute_metrics(y_te, y_pred_rule, y_pred_rule.astype(float), "Rule-based baseline")

# ---------------------------------------------------------------------
# 4. Random Forest, Isolation Forest, LSTM (each trained ONCE on the
#    aligned split; reused both standalone and inside the hybrid ensemble)
# ---------------------------------------------------------------------
t0 = time.time()
rf = mdl.build_random_forest()
rf.fit(Xs_tr_res, y_tr_res)
p_rf = rf.predict_proba(Xs_te_scaled)[:, 1]
m_rf = ev.compute_metrics(y_te, (p_rf >= 0.5).astype(int), p_rf, "Random Forest")
print(f"[3] RF trained in {time.time()-t0:.1f}s")

t0 = time.time()
iforest = mdl.build_isolation_forest()
iforest.fit(Xs_tr_scaled[y_tr == 0])
p_if = mdl.isolation_forest_scores(iforest, Xs_te_scaled)
m_if = ev.compute_metrics(y_te, (p_if >= 0.5).astype(int), p_if, "Isolation Forest")
print(f"[4] IsolationForest trained in {time.time()-t0:.1f}s")

t0 = time.time()
lstm = mdl.build_lstm(Xq_tr.shape[1], Xq_tr.shape[2])
n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}
lstm.fit(Xq_tr, y_tr, epochs=LSTM_EPOCHS, batch_size=512, validation_split=0.1,
         class_weight=class_weight, verbose=2)
p_lstm = lstm.predict(Xq_te, verbose=0).ravel()
m_lstm = ev.compute_metrics(y_te, (p_lstm >= 0.5).astype(int), p_lstm, "LSTM")
print(f"[5] LSTM trained in {time.time()-t0:.1f}s (class_weight={class_weight})")

# Hybrid soft-voting ensemble. Weights 0.4 / 0.3 / 0.3 are fixed a priori from
# the approved methodology (Section 3.6); they are NOT tuned on any data.
p_hybrid = 0.4 * p_rf + 0.3 * p_if + 0.3 * p_lstm
m_hybrid = ev.compute_metrics(y_te, (p_hybrid >= 0.5).astype(int), p_hybrid, "Hybrid Ensemble")

results_42 = ev.results_table([m_rule, m_rf, m_if, m_lstm, m_hybrid])
results_42.to_csv("results/4_2_model_comparison.csv")
print("\n=== 4.2 Model comparison (RQ1) ===")
print(results_42)

# --- Figure 4.1: F1-score comparison across models ---
fig, ax = plt.subplots(figsize=(7, 4.5))
model_names = results_42.index.tolist()
f1_vals = results_42["f1"].tolist()
bars = ax.bar(model_names, f1_vals, color=["#9e9e9e", "#4c72b0", "#dd8452", "#55a868", "#c44e52"])
ax.set_ylabel("F1-score")
ax.set_title("Figure 4.1: F1-Score by Model (IAM-SynthSim Regime A Test Set)")
ax.set_ylim(0, 1.0)
for b, v in zip(bars, f1_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("results/figures/fig4_1_f1_comparison.png", dpi=150)
plt.close()

# --- Figure 4.2: Confusion matrix (Hybrid Ensemble) + ROC curves ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
cm = confusion_matrix(y_te, (p_hybrid >= 0.5).astype(int), labels=[0, 1])
im = axes[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Normal", "Threat"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Normal", "Threat"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title("(a) Hybrid Ensemble Confusion Matrix")

for name, proba, y_true_arr in [
    ("Random Forest", p_rf, y_te), ("Isolation Forest", p_if, y_te),
    ("LSTM", p_lstm, y_te), ("Hybrid Ensemble", p_hybrid, y_te),
]:
    fpr, tpr, _ = roc_curve(y_true_arr, proba)
    auc_val = roc_auc_score(y_true_arr, proba)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("(b) ROC Curves")
axes[1].legend(fontsize=8, loc="lower right")
plt.suptitle("Figure 4.2: Hybrid Ensemble Confusion Matrix and Model ROC Curves")
plt.tight_layout()
plt.savefig("results/figures/fig4_2_confusion_roc.png", dpi=150)
plt.close()
print("Saved Figure 4.1 and 4.2")

# ---------------------------------------------------------------------
# 5. Temporal ablation (RQ2): sequential LSTM vs flattened last-event RF
# ---------------------------------------------------------------------
t0 = time.time()
X_flat_tr = Xq_tr[:, -1, :]
X_flat_te = Xq_te[:, -1, :]
rf_flat = mdl.build_random_forest()
rf_flat.fit(X_flat_tr, y_tr)
p_flat = rf_flat.predict_proba(X_flat_te)[:, 1]
m_flat = ev.compute_metrics(y_te, (p_flat >= 0.5).astype(int), p_flat, "Flattened (static-only)")
m_seq = ev.compute_metrics(y_te, (p_lstm >= 0.5).astype(int), p_lstm, "Sequential (LSTM)")

ablation = ev.results_table([m_flat, m_seq])
ablation.to_csv("results/4_3_temporal_ablation.csv")
print(f"\n=== 4.3 Temporal ablation (RQ2), built in {time.time()-t0:.1f}s ===")
print(ablation)

# --- Figure 4.3: Temporal ablation comparison ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
metrics_to_plot = ["precision", "recall", "f1", "auc_roc"]
x = np.arange(len(metrics_to_plot))
width = 0.35
flat_vals = [ablation.loc["Flattened (static-only)", m] for m in metrics_to_plot]
seq_vals = [ablation.loc["Sequential (LSTM)", m] for m in metrics_to_plot]
ax.bar(x - width/2, flat_vals, width, label="Flattened (static-only)", color="#4c72b0")
ax.bar(x + width/2, seq_vals, width, label="Sequential (LSTM)", color="#c44e52")
ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1", "AUC-ROC"])
ax.set_ylim(0, 1.05)
ax.set_title("Figure 4.3: Temporal Ablation — Flattened vs. Sequential (RQ2)")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/fig4_3_temporal_ablation.png", dpi=150)
plt.close()
print("Saved Figure 4.3")

# ---------------------------------------------------------------------
# 6. RQ3 proxy: domain-shift generalization across simulation regimes
#    (Regime A <-> Regime B) using the Random Forest component.
#    NOTE: this is a same-simulator, different-parameterization proxy for
#    cross-dataset generalization, NOT a true second real-world dataset.
# ---------------------------------------------------------------------
def rf_train_test(train_feat, test_feat, same_regime=False):
    # Same-regime: evaluate on the held-out 15% test partition of that regime
    #   (never seen during training). Cross-regime: evaluate on all events of
    #   the other regime (none of which were used in training).
    Xtr_full = train_feat[fe.STATIC_FEATURES].values
    ytr_full = train_feat["label"].values

    Xtr_tr, Xtr_val, Xtr_te2, ytr_tr, ytr_val, ytr_te2 = pp.stratified_split(Xtr_full, ytr_full)
    if same_regime:
        Xte_full, yte_full = Xtr_te2, ytr_te2
    else:
        Xte_full = test_feat[fe.STATIC_FEATURES].values
        yte_full = test_feat["label"].values
    sc = MinMaxScaler().fit(Xtr_tr)
    Xtr_s = sc.transform(Xtr_tr)
    Xtr_res, ytr_res = pp.resample_training_set(Xtr_s, ytr_tr)

    model = mdl.build_random_forest()
    model.fit(Xtr_res, ytr_res)

    Xte_s = sc.transform(Xte_full)
    p = model.predict_proba(Xte_s)[:, 1]
    return ev.compute_metrics(yte_full, (p >= 0.5).astype(int), p, "x")

t0 = time.time()
same_a = rf_train_test(feat_a, feat_a, same_regime=True)
cross_a_to_b = rf_train_test(feat_a, feat_b)
same_b = rf_train_test(feat_b, feat_b, same_regime=True)
cross_b_to_a = rf_train_test(feat_b, feat_a)

rq3_table = pd.DataFrame([
    {"setting": "Train Regime A / Test Regime A (same)", **same_a},
    {"setting": "Train Regime A / Test Regime B (cross)", **cross_a_to_b},
    {"setting": "Train Regime B / Test Regime B (same)", **same_b},
    {"setting": "Train Regime B / Test Regime A (cross)", **cross_b_to_a},
]).set_index("setting").round(4)
rq3_table.to_csv("results/4_4_cross_regime.csv")
print(f"\n=== 4.4 Cross-regime generalization proxy (RQ3), built in {time.time()-t0:.1f}s ===")
print(rq3_table)

# --- Figure 4.4: Cross-regime generalization bar chart ---
fig, ax = plt.subplots(figsize=(7.5, 4.5))
settings = rq3_table.index.tolist()
f1_vals = rq3_table["f1"].tolist()
colors = ["#4c72b0", "#c44e52", "#4c72b0", "#c44e52"]
bars = ax.bar(range(len(settings)), f1_vals, color=colors)
ax.set_xticks(range(len(settings)))
ax.set_xticklabels(["Train A/\nTest A", "Train A/\nTest B", "Train B/\nTest B", "Train B/\nTest A"])
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1.05)
ax.set_title("Figure 4.4: Cross-Regime Generalization (RQ3 Proxy)")
for b, v in zip(bars, f1_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/fig4_4_cross_regime.png", dpi=150)
plt.close()
print("Saved Figure 4.4")

# ---------------------------------------------------------------------
# 7. SHAP feature importance (Random Forest component)
# ---------------------------------------------------------------------
t0 = time.time()
importance = ev.shap_feature_importance(rf, Xs_te_scaled, fe.STATIC_FEATURES, max_display=6)
importance.to_csv("results/4_5_feature_importance.csv", index=False)
print(f"\n=== 4.5 SHAP feature importance, built in {time.time()-t0:.1f}s ===")
print(importance)

# --- Figure 4.5: SHAP feature importance bar chart ---
fig, ax = plt.subplots(figsize=(7, 4.5))
feats = importance["feature"].tolist()[::-1]
vals = importance["mean_abs_shap"].tolist()[::-1]
ax.barh(feats, vals, color="#4c72b0")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Figure 4.5: SHAP Feature Importance (Random Forest, Regime A)")
plt.tight_layout()
plt.savefig("results/figures/fig4_5_shap_importance.png", dpi=150)
plt.close()
print("Saved Figure 4.5")

print("\nALL DONE")
