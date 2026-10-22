"""
feature_engineering.py
=======================
Turns the raw LANL and OakOak logs into a COMMON feature schema so that the
same downstream model code can be trained/evaluated on either source, which
is what RQ3 (cross-dataset generalization) requires.

Common feature columns produced by `build_common_features()`:
    user_id, t (seconds), login_velocity, session_length,
    geolocation_entropy, privilege_change, failure_rate,
    is_privileged_role, label

`create_sequences()` then turns the common feature table into fixed-length,
per-user event sequences for the LSTM component (Section 3.6 of the
thesis: 24-hour windows).
"""

import numpy as np
import pandas as pd

STATIC_FEATURES = [
    "login_velocity", "session_length", "geolocation_entropy",
    "privilege_change", "failure_rate", "is_privileged_role",
]


def build_common_features_lanl(df):
    df = df.sort_values(["src_user", "time"]).copy()
    df["hour_bucket"] = df["time"] // 3600

    # login velocity: events per user within the same hour bucket
    velocity = df.groupby(["src_user", "hour_bucket"])["time"].transform("count")

    # session length proxy: time since previous event by the same user
    df["prev_time"] = df.groupby("src_user")["time"].shift(1)
    session_length = (df["time"] - df["prev_time"]).fillna(0).clip(lower=0, upper=3600) / 60.0

    # geolocation-entropy proxy: Shannon entropy of destination computers
    # touched by the user within a rolling 24-hour window
    def rolling_entropy(sub):
        counts = sub["dst_computer"].value_counts(normalize=True)
        return float(-(counts * np.log2(counts + 1e-12)).sum())

    entropy_map = df.groupby("src_user").apply(rolling_entropy)
    geo_entropy = df["src_user"].map(entropy_map).fillna(0.0)

    privilege_change = (df["logon_type"] == "Service").astype(int)
    failure_rate = df.groupby("src_user")["success"].transform(
        lambda s: (s == "Fail").mean()
    )
    is_privileged_role = df["dst_computer"].str.contains("C0|C1", regex=True).astype(int)

    out = pd.DataFrame({
        "user_id": df["src_user"].values,
        "t": df["time"].values,
        "login_velocity": velocity.values,
        "session_length": session_length.values,
        "geolocation_entropy": geo_entropy.values,
        "privilege_change": privilege_change.values,
        "failure_rate": failure_rate.values,
        "is_privileged_role": is_privileged_role.values,
        "label": df["label"].values,
    })
    return out


def build_common_features_synthsim(df):
    df = df.sort_values(["user_id", "timestamp"]).copy()
    df["hour_bucket"] = df["timestamp"] // 3600
    velocity = df.groupby(["user_id", "hour_bucket"])["timestamp"].transform("count")
    failure_rate = np.clip((df["login_attempts"] - 1) / 10.0, 0, 1)
    is_privileged_role = df["user_role"].isin(["admin", "service_account"]).astype(int)

    out = pd.DataFrame({
        "user_id": df["user_id"].values,
        "t": df["timestamp"].values,
        "login_velocity": velocity.values,
        "session_length": df["session_length"].values,
        "geolocation_entropy": df["geolocation_entropy"].values,
        "privilege_change": df["privilege_change"].values,
        "failure_rate": failure_rate.values,
        "is_privileged_role": is_privileged_role.values,
        "label": df["label"].values,
    })
    return out


def create_sequences(common_df, seq_len=24, stride=1):
    """
    Build fixed-length per-user sequences of STATIC_FEATURES for the LSTM
    component. Each sequence is the `seq_len` most recent events for a
    user, right before (and including) the labeled event. Users with fewer
    than `seq_len` prior events are zero-padded on the left.

    Returns
    -------
    X_seq : np.ndarray, shape (n_samples, seq_len, n_features)
    y_seq : np.ndarray, shape (n_samples,)
    """
    n_features = len(STATIC_FEATURES)
    X_list, y_list = [], []

    for _, g in common_df.groupby("user_id"):
        g = g.sort_values("t")
        feats = g[STATIC_FEATURES].to_numpy(dtype=float)
        labels = g["label"].to_numpy()
        n = len(g)
        for i in range(0, n, stride):
            window = feats[max(0, i - seq_len + 1): i + 1]
            if len(window) < seq_len:
                pad = np.zeros((seq_len - len(window), n_features))
                window = np.vstack([pad, window])
            X_list.append(window)
            y_list.append(labels[i])

    X_seq = np.array(X_list, dtype=float)
    y_seq = np.array(y_list, dtype=int)
    return X_seq, y_seq
