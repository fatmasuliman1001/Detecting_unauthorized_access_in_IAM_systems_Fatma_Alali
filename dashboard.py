"""
dashboard.py
============
Interactive Streamlit monitoring dashboard for the hybrid IAM unauthorized-access
detection framework described in Chapter 3 of the thesis (see Appendix B).

Run with:
    streamlit run dashboard.py

It reads the real result CSVs produced by run_real_experiment.py (results/*.csv)
and the figures produced alongside them (results/figures/*.png); nothing on this
dashboard is hard-coded, it visualizes exactly what that script computed. If results/ is empty, run `python run_real_experiment.py` first.
"""
import os
import pandas as pd
import streamlit as st

# Locate results/ relative to this file, so the dashboard works from any folder.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

st.set_page_config(page_title="IAM Unauthorized-Access Detection Dashboard", layout="wide")

st.title("Hybrid IAM Unauthorized-Access Detection — Monitoring Dashboard")
st.caption(
    "Reads real output from run_real_experiment.py. Results shown here reflect "
    "IAM-SynthSim (a custom synthetic simulator), not real enterprise traffic -- "
    "see Chapter 3 of the thesis for a description of the data."
)


def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        st.warning(f"{name} not found yet. Run `python run_real_experiment.py` first.")
        return None
    return pd.read_csv(path, index_col=0)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "4.1 Class Balance", "4.2 Model Comparison (RQ1)",
    "4.3 Temporal Ablation (RQ2)", "4.4 Cross-Regime (RQ3)",
    "4.5 Feature Importance",
])

with tab1:
    st.subheader("Class Balance: IAM-SynthSim Regimes A and B")
    df = load_csv("4_1_class_balance.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["n_threat_events"])

with tab2:
    st.subheader("Model Comparison on Regime A Held-Out Test Set")
    df = load_csv("4_2_model_comparison.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df["f1"])
            st.caption("F1-score by model")
        with col2:
            st.bar_chart(df[["precision", "recall"]])
            st.caption("Precision vs. recall by model")
        fig_path = os.path.join(FIG_DIR, "fig4_2_confusion_roc.png")
        if os.path.exists(fig_path):
            st.image(fig_path, caption="Figure 4.2: Confusion matrix and ROC curves", use_container_width=True)
        st.metric("Hybrid Ensemble false-negative rate", f"{df.loc['Hybrid Ensemble', 'false_negative_rate']:.3f}"
                   if "Hybrid Ensemble" in df.index else "n/a")

with tab3:
    st.subheader("Temporal Ablation: Sequential (LSTM) vs. Flattened")
    df = load_csv("4_3_temporal_ablation.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["f1"])
        st.info(
            "This thesis's own run found the flattened representation outperformed the "
            "sequential LSTM (see Chapter 5, Section 5.2), contrary to the literature's "
            "general expectation."
        )

with tab4:
    st.subheader("Cross-Regime Generalization (RQ3 Proxy)")
    df = load_csv("4_4_cross_regime.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["f1"])

with tab5:
    st.subheader("SHAP Feature Importance (Random Forest, Regime A)")
    df = load_csv("4_5_feature_importance.csv")
    if df is not None:
        df = df.set_index("feature") if "feature" in df.columns else df
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.iloc[:, 0])

st.divider()
st.caption(
    "Built for the thesis 'Predictive Analytics for Detecting Unauthorized Access in IAM Systems'. "
    "Source: dashboard.py, part of the accompanying codebase (Appendix B)."
)
