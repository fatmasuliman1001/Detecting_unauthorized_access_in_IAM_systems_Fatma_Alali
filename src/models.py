"""
models.py
=========
Implements the three components of the hybrid model (Section 3.6):

  - Random Forest (supervised)         : n_estimators=500, max_depth=15
  - Isolation Forest (unsupervised)     : n_estimators=200, contamination=0.01
  - Bidirectional LSTM (sequential)     : 2 layers, 128 units, dropout=0.2

and combines them with a soft-voting ensemble weighted
(RF: 0.4, iForest: 0.3, LSTM: 0.3), as specified in the thesis.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as kmodels
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_random_forest(random_state=42):
    return RandomForestClassifier(
        n_estimators=500, max_depth=15, class_weight="balanced_subsample",
        random_state=random_state, n_jobs=-1,
    )


def build_isolation_forest(random_state=42):
    return IsolationForest(
        n_estimators=200, contamination=0.01, random_state=random_state, n_jobs=-1,
    )


def isolation_forest_scores(iforest, X):
    """Convert IsolationForest's raw score (higher = more normal) into a
    pseudo-probability of being anomalous, scaled to [0, 1]."""
    raw = -iforest.score_samples(X)  # higher = more anomalous
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)


def build_lstm(seq_len, n_features, units=128, dropout=0.2):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for the LSTM component.")
    model = kmodels.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.Bidirectional(layers.LSTM(units, return_sequences=True, dropout=dropout)),
        layers.Bidirectional(layers.LSTM(units, dropout=dropout)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


class HybridEnsemble:
    """
    Soft-voting ensemble of Random Forest, Isolation Forest, and LSTM,
    as specified in Section 3.6: weights RF=0.4, iForest=0.3, LSTM=0.3.
    """

    def __init__(self, weights=(0.4, 0.3, 0.3), random_state=42):
        self.w_rf, self.w_if, self.w_lstm = weights
        self.rf = build_random_forest(random_state)
        self.iforest = build_isolation_forest(random_state)
        self.lstm = None
        self.random_state = random_state

    def fit(self, X_static, y_static, X_seq, y_seq, epochs=10, batch_size=256, verbose=0):
        self.rf.fit(X_static, y_static)
        self.iforest.fit(X_static[y_static == 0])  # learn "normal" manifold
        self.lstm = build_lstm(X_seq.shape[1], X_seq.shape[2])
        n_pos, n_neg = int(y_seq.sum()), int((y_seq == 0).sum())
        class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}
        # validation_split only logs val-loss per epoch; no early stopping / selection.
        self.lstm.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size,
                      validation_split=0.1, class_weight=class_weight, verbose=verbose)
        return self

    def predict_proba(self, X_static, X_seq):
        p_rf = self.rf.predict_proba(X_static)[:, 1]
        p_if = isolation_forest_scores(self.iforest, X_static)
        p_lstm = self.lstm.predict(X_seq, verbose=0).ravel()
        return self.w_rf * p_rf + self.w_if * p_if + self.w_lstm * p_lstm

    def predict(self, X_static, X_seq, threshold=0.5):
        return (self.predict_proba(X_static, X_seq) >= threshold).astype(int)
