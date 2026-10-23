# Hybrid IAM Unauthorized-Access Detection — Code Companion to Chapters 3–5

This is the runnable implementation of the methodology in **Chapter 3** of the thesis
*"Predictive Analytics for Detecting Unauthorized Access in Identity and Access Management
Systems."* **It has already been run, and its real output is what Chapters 4 and 5 report.**

## Quick start (reproduce all results)

Requires **Python 3.11** (the results were produced with Python 3.11.9; much newer Python versions
may not support the pinned TensorFlow version). Takes about 5–10 minutes on a normal laptop CPU.

**1. Get the code:** click the green **Code** button on this page → **Download ZIP**, then unzip it
(or run `git clone` with this repository's URL).

**2. Open a terminal inside the unzipped folder** and run:

macOS / Linux:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_real_experiment.py
```

Windows (Command Prompt):
```bat
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_real_experiment.py
```

**3. Check the results:** the script regenerates `data/` (the two simulated datasets) and
`results/` (Tables 4.1–4.5 as CSV, Figures 4.1–4.5 as PNG). They should match the files already
in this repository, which are the exact values reported in Chapter 4 of the thesis.

**4. (Optional) View the interactive dashboard:**
```bash
streamlit run dashboard.py
```
It opens in your browser at http://localhost:8501.

The scripts can also be run from an IDE (PyCharm, VS Code, Cursor): open this folder, select the
`venv` interpreter, and run `run_real_experiment.py`.

## Status: real results obtained, with a disclosed scope limitation

Two things from the original proposal didn't hold up:

1. **LANL Cyber1** could not be downloaded from the sandboxed environment used to build and run
   this pipeline — its network access is restricted to a fixed allowlist (PyPI, npm, GitHub, etc.)
   that does not include `csr.lanl.gov` (confirmed directly: the host returns `host_not_allowed`).
2. **"OakOak"**, the secondary dataset named in the proposal, could not be verified as a real,
   documented tool after a systematic search. No repository, paper, or product by that name could
   be found.

Rather than fabricate results under either name, this codebase uses **IAM-SynthSim** — a
custom, fully-specified, agent-based synthetic IAM log simulator built for this study and
described transparently as such in Chapter 3. It's run under two parameterizations (**Regime A**
and **Regime B**) that stand in for different deployment environments, used as a proxy for the
cross-environment generalization question (RQ3) in the absence of a second independent dataset.

**The numbers in Chapter 4 are real** — actually computed by running this code, not invented —
but they describe a small-scale synthetic simulation, not real enterprise traffic. Chapters 4–5
say this explicitly and interpret the results accordingly.

## 1. Project structure

```
iam_project/
├── run_real_experiment.py        ← the script that produced every number in Chapter 4
├── src/
│   ├── data_loader.py            ← loads LANL Cyber1 (real) + IAM-SynthSim (custom simulator)
│   ├── feature_engineering.py    ← common feature schema + LSTM sequence builder
│   ├── preprocessing.py          ← 70/15/15 helper split (RQ3 only), MinMax scaling, SMOTE
│   ├── models.py                 ← Random Forest, Isolation Forest, LSTM, Hybrid Ensemble
│   └── evaluate.py               ← metrics, confusion matrix/ROC plots, SHAP importance
├── results/                      ← the actual CSV outputs reported in Chapter 4
│   ├── 4_1_class_balance.csv
│   ├── 4_2_model_comparison.csv       (RQ1: hybrid vs. baselines)
│   ├── 4_3_temporal_ablation.csv      (RQ2: sequential vs. flattened)
│   ├── 4_4_cross_regime.csv           (RQ3: cross-environment proxy)
│   └── 4_5_feature_importance.csv     (SHAP)
├── data/                         ← the exact simulated datasets used for the thesis results
│   ├── iam_synthsim_regime_A.csv      (30,000 events, 300 threats)
│   └── iam_synthsim_regime_B.csv      (30,000 events, 300 threats)
└── README.md
```

## 2. What the real run found (summary — see Chapter 4/5 for full discussion)

- **RQ1**: Random Forest alone was the strongest single component (F1 = 0.886, AUC = 0.999). The
  hybrid ensemble matched or outperformed it on every reported metric, achieving perfect recall
  (1.000, zero missed threats), precision of 0.818 and F1 = 0.900. The fixed 0.4/0.3/0.3
  soft-voting weights from the proposal were used as-is rather than tuned, which is flagged as a
  concrete next step.
- **RQ2 (unexpected)**: the sequential LSTM *underperformed* a flattened, non-sequential feature
  representation (F1 = 0.109 vs. 0.891) — the opposite of what the literature predicts. Most
  likely cause: only 4 training epochs (CPU-only sandbox, bounded execution time) and a small
  minority-class sample (210 labeled threats in the training split). This is reported honestly as
  an inconclusive/contradicted result requiring re-testing with more compute, not hidden or
  reframed as a success.
- **RQ3**: a real, asymmetric cross-regime generalization gap was found — training on Regime A and
  testing on Regime B collapsed F1 from 0.900 to 0.229, while the reverse direction held up much
  better. This is discussed as a genuine, if partial, finding — genuine because it's a real
  computed result, partial because it's a same-simulator proxy, not two independent datasets.
- **SHAP**: geolocation entropy and login-failure rate were the strongest predictors of
  unauthorized access; login velocity was the weakest.

## 3. Re-running it yourself

**Simulated datasets:** the real datasets named in the proposal were not publicly accessible, so all
results come from IAM-SynthSim. The exact simulated data used for every result is provided in `data/`
(`iam_synthsim_regime_A.csv` and `iam_synthsim_regime_B.csv`). Running `run_real_experiment.py`
regenerates these files identically (fixed seed 42) together with everything in `results/`.

**Reproducibility:** the results in `results/` were produced with Python 3.11.9 on macOS, using the
exact library versions pinned in `requirements.txt` and a fixed random seed (42). Re-running on
different hardware or library versions may produce minor variation in the third decimal place
without affecting the conclusions.

```bash
pip install -r requirements.txt
python run_real_experiment.py
```

This regenerates everything in `results/` from scratch (takes several minutes on CPU, mostly
LSTM training).

## 4. How to get REAL LANL Cyber1 results (the essential next step — see Section 6.3)

`src/data_loader.py` already fully supports the real LANL file format. To use it:

1. On a machine **without** this sandbox's network restrictions, go to
   https://csr.lanl.gov/data/cyber1/ and download `auth.txt.gz` and `redteam.txt.gz`.
2. Place both in `data/`.
3. In `run_real_experiment.py`, replace the
   `dl.load_iam_synthsim(...)` calls with:
   ```python
   dl.load_lanl_cyber1(path="data/auth.txt.gz", redteam_path="data/redteam.txt.gz", n_rows=2_000_000)
   ```
4. Re-run. The same feature engineering, models, and evaluation code will now produce results on
   real enterprise authentication data, which is what should ultimately anchor Chapters 4–5 rather
   than the synthetic-simulator results currently there.

If you have access to whatever tool your mentor actually intended by "OakOak," `load_iam_synthsim()`
also accepts a `path=` argument pointing at any CSV with the documented column schema — swap it in
the same way.

## 5. Compute notes

- The LSTM component (2-layer, 128-unit bidirectional LSTM) is the slowest part of the pipeline on
  CPU (~45s/epoch at 30,000 events). The results in `results/` used only 4 epochs because of the
  sandbox's execution-time limits — Chapter 5 discusses why this likely explains the RQ2 result and
  recommends substantially more epochs (and GPU hardware, as originally specified) for the
  confirmatory real-data run.
- `preprocessing.resample_training_set()` applies SMOTE to the **training split only** — the test
  set keeps the real 1:100 class ratio, so reported metrics reflect realistic conditions.
- **Which split is which:** the main experiment (`run_real_experiment.py`; Tables 4.2/4.3, Figs 4.1–4.3)
  uses one stratified **70/30 train/test** split (random_state=42) with no validation subset, and the
  ensemble weights 0.4/0.3/0.3 are fixed a priori, not tuned. `preprocessing.stratified_split()` is a
  separate **70/15/15** helper used only by the RQ3 cross-regime analysis (Section 4.4).

## 6. What's intentionally not included

- **Streamlit dashboard**: now included as `dashboard.py` (run `streamlit run dashboard.py`; see Appendix C).
- **Live/streaming ingestion**: the pipeline is batch/offline, consistent with the study's disclosed scope and limitations (Section 1.5).

## 7. Citation

If you use the LANL dataset, cite: Kent, A. D. (2015). *Comprehensive, multi-source
cyber-security events data set.* Los Alamos National Laboratory. https://doi.org/10.17021/1179829
