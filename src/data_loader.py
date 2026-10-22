"""
data_loader.py
==============
Loads the two data sources described in Chapter 3 of the thesis:

  1. LANL Cyber1 ("Comprehensive, Multi-Source Cyber-Security Events" dataset,
     Kent, 2015) - real enterprise authentication logs.
     Official source: https://csr.lanl.gov/data/cyber1/
     File used: auth.txt.gz  (~50 GB uncompressed for the full file; a
     truncated number of rows is usually enough for a thesis-scale sample).

  2. OakOak-style synthetic IAM logs - agent-based simulated logs with
     ground-truth threat labels (brute_force, lateral_movement,
     privilege_abuse).

IMPORTANT
---------
This sandbox has no network access to csr.lanl.gov, so the real LANL file
cannot be downloaded from inside this environment. `load_lanl_cyber1()`
will therefore:
  - use a real local copy of auth.txt / auth.txt.gz if you provide `path`
  - otherwise fall back to `generate_synthetic_lanl_like()`, which produces
    data with the SAME COLUMN SCHEMA as the real file, purely so the rest
    of the pipeline (feature engineering, models, evaluation) can be
    exercised end-to-end before you have the real data.

Any numbers produced from the synthetic fallback are NOT real results and
must not be reported in Chapter 4/5 of the thesis. Once you download the
real LANL file (see README.md) and point `path` at it, the exact same code
will run on real data and produce real, reportable results.
"""

import gzip
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

LANL_AUTH_COLUMNS = [
    "time", "src_user", "dst_user", "src_computer", "dst_computer",
    "auth_type", "logon_type", "auth_orientation", "success",
]


# ---------------------------------------------------------------------------
# 1. LANL Cyber1
# ---------------------------------------------------------------------------
def load_lanl_cyber1(path=None, n_rows=None, redteam_path=None):
    """
    Load the LANL Cyber1 auth.txt(.gz) file.

    Parameters
    ----------
    path : str or None
        Path to a real `auth.txt` or `auth.txt.gz` file downloaded from
        https://csr.lanl.gov/data/cyber1/. If None, synthetic demo data
        with the same schema is generated instead (see module docstring).
    n_rows : int or None
        If set, only read the first n_rows (useful for the ~1 billion-row
        full file; a few million rows is normally enough for a thesis
        sample).
    redteam_path : str or None
        Path to the accompanying `redteam.txt(.gz)` file, which contains
        the ground-truth compromise events used as labels. If None and
        `path` is also None, synthetic labels are generated instead.

    Returns
    -------
    pandas.DataFrame with columns:
        time, src_user, dst_user, src_computer, dst_computer, auth_type,
        logon_type, auth_orientation, success, label (1 = malicious/insider,
        0 = normal)
    """
    if path is None:
        n = n_rows or 200_000
        return _generate_synthetic_lanl_like(n_events=n)

    opener = gzip.open if str(path).endswith(".gz") else open
    df = pd.read_csv(
        opener(path, "rt"),
        names=LANL_AUTH_COLUMNS,
        nrows=n_rows,
        header=None,
    )
    df["label"] = 0
    if redteam_path is not None:
        rt_opener = gzip.open if str(redteam_path).endswith(".gz") else open
        redteam = pd.read_csv(
            rt_opener(redteam_path, "rt"),
            names=["time", "user", "src_computer", "dst_computer"],
            header=None,
        )
        malicious_keys = set(
            zip(redteam["time"], redteam["user"], redteam["src_computer"], redteam["dst_computer"])
        )
        key = list(zip(df["time"], df["src_user"], df["src_computer"], df["dst_computer"]))
        df["label"] = [1 if k in malicious_keys else 0 for k in key]
    return df


def _generate_synthetic_lanl_like(n_events=200_000, anomaly_rate=0.001):
    """
    SYNTHETIC DEMO DATA ONLY - matches the LANL auth.txt schema so the rest
    of the pipeline can be developed/tested before the real file is
    available. Do not use these numbers in Chapter 4/5.
    """
    n_users = max(200, n_events // 500)
    n_computers = max(50, n_events // 2000)
    users = [f"U{i}@DOM1" for i in range(n_users)]
    computers = [f"C{i}" for i in range(n_computers)]
    auth_types = ["Kerberos", "NTLM", "Negotiate"]
    logon_types = ["Network", "Interactive", "Service", "Batch"]
    orientations = ["LogOn", "LogOff"]

    time = np.sort(RNG.integers(0, 58 * 24 * 3600, size=n_events))
    src_user = RNG.choice(users, n_events)
    dst_user = RNG.choice(users, n_events)
    src_computer = RNG.choice(computers, n_events)
    dst_computer = RNG.choice(computers, n_events)
    auth_type = RNG.choice(auth_types, n_events, p=[0.6, 0.3, 0.1])
    logon_type = RNG.choice(logon_types, n_events, p=[0.5, 0.3, 0.15, 0.05])
    auth_orientation = RNG.choice(orientations, n_events, p=[0.55, 0.45])
    success = RNG.choice(["Success", "Fail"], n_events, p=[0.95, 0.05])

    label = (RNG.random(n_events) < anomaly_rate).astype(int)
    # Make "malicious" rows look structurally different (more distinct
    # dst_computers touched in quick succession, higher failure rate) so
    # that a model trained on this demo data has something learnable.
    fail_idx = np.where(label == 1)[0]
    if len(fail_idx) > 0:
        flip = RNG.random(len(fail_idx)) < 0.5
        success = success.copy()
        success[fail_idx[flip]] = "Fail"

    df = pd.DataFrame({
        "time": time, "src_user": src_user, "dst_user": dst_user,
        "src_computer": src_computer, "dst_computer": dst_computer,
        "auth_type": auth_type, "logon_type": logon_type,
        "auth_orientation": auth_orientation, "success": success,
        "label": label,
    })
    return df


# ---------------------------------------------------------------------------
# 2. IAM-SynthSim: custom agent-based synthetic IAM log simulator
# ---------------------------------------------------------------------------
def load_iam_synthsim(path=None, n_events=200_000, regime="A"):
    """
    Load (or generate) IAM-SynthSim logs: a custom, agent-based synthetic
    IAM log simulator built for this study (see Section 3.4) with
    ground-truth threat labels. This is NOT a pre-existing third-party
    dataset ("OakOak") - no such publicly documented tool could be
    located, so this thesis instead uses its own, fully-specified
    generator, described transparently as synthetic.

    If `path` (a CSV) is provided it must contain at minimum:
    timestamp, user_id, login_attempts, privilege_change, session_length,
    geolocation_entropy, user_role, threat_type, label.

    `regime` selects between two parameterizations ("A" and "B") of the
    same generative process, used as a domain-shift proxy for RQ3 in the
    absence of a second real-world dataset (see Chapter 3 note on scope).
    """
    if path is not None:
        return pd.read_csv(path)
    return _generate_iam_synthsim(n_events=n_events, regime=regime)


def _generate_iam_synthsim(n_events=200_000, threat_rate=0.01, regime="A"):
    # Regime "B" shifts the generative parameters (role mix, session-length
    # distribution, threat composition) to emulate a structurally different
    # deployment environment, without claiming to be a second real dataset.
    if regime == "B":
        role_p = [0.60, 0.25, 0.10, 0.05]
        session_mu, session_sigma = 40, 22
        geo_mu, geo_sigma = 0.5, 0.3
    else:
        role_p = [0.75, 0.15, 0.07, 0.03]
        session_mu, session_sigma = 25, 15
        geo_mu, geo_sigma = 0.3, 0.2

    n_users = max(300, n_events // 300)
    users = [f"user_{i}" for i in range(n_users)]
    roles = ["employee", "contractor", "admin", "service_account"]

    timestamp = np.sort(RNG.integers(0, 90 * 24 * 3600, size=n_events))
    user_id = RNG.choice(users, n_events)
    user_role = RNG.choice(roles, n_events, p=role_p)
    login_attempts = RNG.poisson(1.2, n_events) + 1
    session_length = np.abs(RNG.normal(session_mu, session_sigma, n_events))  # minutes
    geolocation_entropy = np.abs(RNG.normal(geo_mu, geo_sigma, n_events))
    privilege_change = RNG.binomial(1, 0.02, n_events)

    label = np.zeros(n_events, dtype=int)
    threat_type = np.array(["none"] * n_events, dtype=object)

    n_threats = int(n_events * threat_rate)
    threat_idx = RNG.choice(n_events, n_threats, replace=False)
    thirds = np.array_split(threat_idx, 3)

    # brute force: many login attempts, short sessions
    login_attempts[thirds[0]] = RNG.integers(8, 40, len(thirds[0]))
    session_length[thirds[0]] = np.abs(RNG.normal(3, 2, len(thirds[0])))
    threat_type[thirds[0]] = "brute_force"

    # lateral movement: high geolocation entropy, longer sessions
    geolocation_entropy[thirds[1]] = np.abs(RNG.normal(1.5, 0.4, len(thirds[1])))
    session_length[thirds[1]] = np.abs(RNG.normal(60, 20, len(thirds[1])))
    threat_type[thirds[1]] = "lateral_movement"

    # privilege abuse: privilege_change flag set, admin/service roles
    privilege_change[thirds[2]] = 1
    user_role[thirds[2]] = RNG.choice(["admin", "service_account"], len(thirds[2]))
    threat_type[thirds[2]] = "privilege_abuse"

    label[threat_idx] = 1

    df = pd.DataFrame({
        "timestamp": timestamp, "user_id": user_id, "user_role": user_role,
        "login_attempts": login_attempts, "session_length": session_length,
        "geolocation_entropy": geolocation_entropy,
        "privilege_change": privilege_change,
        "threat_type": threat_type, "label": label,
    })
    return df
