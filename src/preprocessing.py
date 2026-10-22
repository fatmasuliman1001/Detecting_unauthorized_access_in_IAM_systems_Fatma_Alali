"""
preprocessing.py
=================
Helper module for Sections 3.3 and 3.5 of the thesis.

NOTE ON SPLIT CONSISTENCY: stratified_split() below implements a 70/15/15
train/validation/test split and is used ONLY by the RQ3 cross-regime
analysis (Section 4.4). It is NOT the split behind Tables 4.2 / 4.3 or
Figures 4.1-4.3 in Chapter 4: those use a direct, stratified 70/30
train/test split with no validation subset (see run_real_experiment.py),
since none of the compared models needs one. The two schemes are
intentionally different and should not be conflated (see Section 3.3).

resample_training_set() applies SMOTE / ADASYN to the training partition
ONLY; evaluation partitions keep the real 1:100 class imbalance.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN


def stratified_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """70/15/15 stratified split, preserving class balance in val/test.
    Used only by the RQ3 cross-regime analysis, not by the main experiment."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def resample_training_set(X_train, y_train, method="smote", target_ratio=0.1,
                           random_state=42):
    """
    Oversample the minority (threat) class in the TRAINING set only.
    target_ratio=0.1 means minority class will be ~10% of majority after
    resampling (a 1:10 ratio), matching Section 3.5. If the minority class
    already exceeds that ratio, no resampling is applied.
    """
    n_majority = (y_train == 0).sum()
    n_minority = (y_train == 1).sum()
    if n_minority == 0 or n_minority / max(n_majority, 1) >= target_ratio:
        return X_train, y_train

    desired_minority = int(n_majority * target_ratio)
    sampler_cls = SMOTE if method == "smote" else ADASYN
    sampler = sampler_cls(
        sampling_strategy={1: max(desired_minority, n_minority + 1)},
        random_state=random_state,
        k_neighbors=min(5, max(1, n_minority - 1)),
    )
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res
