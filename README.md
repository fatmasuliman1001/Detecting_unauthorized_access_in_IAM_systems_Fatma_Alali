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

## Data

All results are produced from **IAM-SynthSim**, a custom, fully specified, agent-based synthetic
IAM log simulator built for this study (described in Chapter 3). Synthetic data is used because
real IAM logs with reliable ground-truth labels for unauthorized access are rarely publicly
available, and a fully specified simulator makes every result reproducible and the class balance
controllable. The simulator is run under two parameterizations (**Regime A** and **Regime B**)
that represent different deployment environments, used to study cross-environment
generalization (RQ3).

The results describe a small-scale synthetic simulation, not real enterprise traffic; Chapters
4–5 interpret them accordingly.

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

## 2. Summary of results (see Chapters 4–5 for full discussion)

- **RQ1**: Random Forest alone was the strongest single component (F1 = 0.886, AUC = 0.999). The
  hybrid ensemble matched or outperformed it on every reported metric, achieving perfect recall
  (1.000, zero missed threats), precision of 0.818 and F1 = 0.900. The 0.4/0.3/0.3 soft-voting
  weights were fixed a priori rather than tuned; tuning them is a concrete next step.
- **RQ2 (unexpected)**: the sequential LSTM *underperformed* a flattened, non-sequential feature
  representation (F1 = 0.109 vs. 0.891) — the opposite of what the literature predicts. Most
  likely cause: only 4 training epochs (CPU-only training) and a small minority-class sample
  (210 labeled threats in the training split). This result is treated as inconclusive and
  requires re-testing with more training.
- **RQ3**: a real, asymmetric cross-regime generalization gap was found — training on Regime A and
  testing on Regime B collapsed F1 from 0.900 to 0.229, while the reverse direction held up much
  better. Because both regimes come from the same simulator, this is a proxy for, not a direct
  test of, generalization across independent datasets.
- **SHAP**: geolocation entropy and login-failure rate were the strongest predictors of
  unauthorized access; login velocity was the weakest.

## 3. Re-running it yourself

**Simulated datasets:** the exact simulated data used for every result is provided in `data/`
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

## 4. Running on real data (future work)

`src/data_loader.py` also supports the public LANL Cyber1 authentication dataset. To use it:

1. Go to https://csr.lanl.gov/data/cyber1/ and download `auth.txt.gz` and `redteam.txt.gz`.
2. Place both in `data/`.
3. In `run_real_experiment.py`, replace the
   `dl.load_iam_synthsim(...)` calls with:
   ```python
   dl.load_lanl_cyber1(path="data/auth.txt.gz", redteam_path="data/redteam.txt.gz", n_rows=2_000_000)
   ```
4. Re-run. The same feature engineering, models, and evaluation code will then produce results on
   real enterprise authentication data.

`load_iam_synthsim()` also accepts a `path=` argument pointing at any CSV with the documented
column schema, so other labelled IAM datasets can be used in the same way.

## 5. Compute notes

- The LSTM component (2-layer, 128-unit bidirectional LSTM) is the slowest part of the pipeline on
  CPU (~45s/epoch at 30,000 events). The results in `results/` used 4 epochs on CPU-only
  hardware; Chapter 5 discusses why this likely explains the RQ2 result and recommends
  substantially more epochs (and GPU hardware) for future runs.
- `preprocessing.resample_training_set()` applies SMOTE to the **training split only** — the test
  set keeps the real 1:100 class ratio, so reported metrics reflect realistic conditions.
- **Which split is which:** the main experiment (`run_real_experiment.py`; Tables 4.2/4.3, Figs 4.1–4.3)
  uses one stratified **70/30 train/test** split (random_state=42) with no validation subset, and the
  ensemble weights 0.4/0.3/0.3 are fixed a priori, not tuned. `preprocessing.stratified_split()` is a
  separate **70/15/15** helper used only by the RQ3 cross-regime analysis (Section 4.4).

## 6. Dashboard and scope

- **Streamlit dashboard**: `dashboard.py` (run `streamlit run dashboard.py`; see Appendix B).
- **Live/streaming ingestion** is not included: the pipeline is batch/offline, consistent with the
  study's scope.

## 7. Citation

If you use the LANL dataset, cite: Kent, A. D. (2015). *Comprehensive, multi-source
cyber-security events data set.* Los Alamos National Laboratory. https://doi.org/10.17021/1179829
