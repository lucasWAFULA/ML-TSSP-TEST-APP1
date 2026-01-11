# ======================================================
# MODE CONFIGURATION
# ======================================================
MODE = "streamlit"  # options: "streamlit", "api", "batch"

# ======================================================
# CORE IMPORTS
# ======================================================
from pathlib import Path
from typing import Dict, List

import numpy as np
import joblib
import shap
from pydantic import BaseModel

# ======================================================
# BASE DIRECTORY & MODEL PATHS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
# Ensure models directory exists (helpful for packaging / first run)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
XGB_MODEL_PATH = MODEL_DIR / "xgb_classifier_reduced.joblib"


# ======================================================
# PART A: BACKEND (FASTAPI / SHARED SCHEMAS)
# ======================================================

class SourceInput(BaseModel):
    source_id: str
    features: Dict[str, float]
    reliability_series: List[float]


class OptimizationRequest(BaseModel):
    sources: List[SourceInput]
    seed: int = 42


class Assignment(BaseModel):
    source_id: str
    task: str
    expected_risk: float


class OptimizationResponse(BaseModel):
    policies: Dict[str, List[Assignment]]
    emv: Dict[str, float]
    evpi: float
    audit_log: Dict[str, object]


# ======================================================
# PART B: ML + SHAP SERVICE
# ======================================================
import logging
logger = logging.getLogger(__name__)
# initialize model variables to avoid NameError in conditional logic
xgb_model = None
explainer = None

if not XGB_MODEL_PATH.exists():
    # Try to locate any candidate .joblib in the models directory
    candidates = list(MODEL_DIR.glob("*.joblib"))
    if candidates:
        XGB_MODEL_PATH = candidates[0]
        logger.warning("XGB model not found at expected path; using %s", XGB_MODEL_PATH)
    else:
        # As a convenience, check the repository root for a dropped model file
        alt = BASE_DIR / "xgb_classifier_reduced.joblib"
        if alt.exists():
            XGB_MODEL_PATH = alt
            logger.warning("Found model at repository root; using %s", XGB_MODEL_PATH)
        else:
            logger.warning(
                "XGBoost model not found at %s. Continuing without model; behavior/prediction features will be disabled.",
                XGB_MODEL_PATH,
            )
            xgb_model = None
            explainer = None

if xgb_model is None and XGB_MODEL_PATH.exists():
    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
    except Exception as e:
        # Specific handling for XGBoost deserialize errors (incompatible/corrupt model)
        logger.exception("Failed to load XGB model at %s: %s", XGB_MODEL_PATH, e)
        # As a helpful fallback for development, provide a tiny dummy model that returns
        # uniform probabilities across expected behavior classes so the UI can still run.
        class _DummyBehaviorModel:
            def __init__(self, classes):
                import numpy as _np
                self.classes_ = _np.array(classes)
            def predict_proba(self, X):
                import numpy as _np
                n = len(self.classes_)
                return _np.ones((X.shape[0], n)) / n

        fallback_classes = ["cooperative", "uncertain", "coerced", "deceptive"]
        logger.error("Using dummy behavior model with classes: %s", fallback_classes)
        xgb_model = _DummyBehaviorModel(fallback_classes)
        # explainer will be constructed in the SHAP service section below
        explainer = None

# ======================================================
# PART C: HELPER FUNCTIONS
# ======================================================

# NOTE: the canonical `explain_source` implementation lives in the
# SHAP service section (below). We intentionally avoid defining a
# different helper with the same name to prevent confusion and
# duplicated warnings during import-time module initialization.
# The SHAP-backed implementation will raise a clear RuntimeError
# if a compatible XGBoost model / explainer is not available.


# ======================================================
# 2. ML Layer
# ======================================================
# i) XGBoost behavior classifier
# ml/xgb_behavior.py

BASE_DIR = Path(__file__).resolve().parent

MODEL_VERSION = "xgb_v4"

XGB_MODEL_PATH = MODEL_DIR / "xgb_classifier_reduced.joblib"
# Default fallback classes (will be overridden if a model with `classes_` is loaded)
BEHAVIOR_CLASSES = [
    "cooperative",
    "uncertain",
    "coerced",
    "deceptive"
]


def predict_behavior_probs(features: dict):
    if xgb_model is None:
        raise RuntimeError(
            "Behavior model unavailable: XGBoost model not loaded. Place a valid model file in 'models/' or set XGB_MODEL_PATH."
        )
    x = np.array(list(features.values())).reshape(1, -1)
    probs = xgb_model.predict_proba(x)[0]
    return dict(zip(BEHAVIOR_CLASSES, probs.tolist()))


# ======================================================
# SHAP service
# ml/shap_explainer.py
# ======================================================

if xgb_model is not None:
    try:
        explainer = shap.TreeExplainer(xgb_model)
    except Exception as e:
        logger.warning("Cannot create SHAP explainer for model %s: %s", type(xgb_model), e)
        explainer = None
else:
    explainer = None

# Feature order expected by the XGBoost model (discovered from the model metadata)
FEATURE_NAMES = [
    "ci_flag",
    "reliability_score",
    "deception_score",
    "scenario_probability",
    "task_success_rate",
    "corroboration_score",
    "handler_confidence",
    "report_timeliness",
]

# If a real model is loaded, derive behavior classes directly from it
if xgb_model is not None and hasattr(xgb_model, "classes_"):
    try:
        BEHAVIOR_CLASSES = [str(c) for c in xgb_model.classes_]
        logger.info("Behavior classes set from model: %s", BEHAVIOR_CLASSES)
    except Exception:
        logger.warning("Failed to read classes_ from loaded XGB model; using fallback BEHAVIOR_CLASSES.")

# Flag whether we are running with a development fallback (dummy) model
MODEL_IS_DUMMY = xgb_model is not None and xgb_model.__class__.__name__.startswith("_DummyBehaviorModel")


def explain_source(features: dict):
    if explainer is None or xgb_model is None:
        raise RuntimeError(
            "SHAP explainer / model unavailable: XGBoost model not loaded. Place a valid model file in 'models/' or set XGB_MODEL_PATH."
        )

    # Build input in the same feature order expected by the model
    x = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)

    shap_values = explainer.shap_values(x)

    explanation = {}

    # Handle multiple shapes returned by different SHAP versions / model types
    try:
        # Case A: shap_values is a list with a single element shaped (n_features, n_classes)
        if isinstance(shap_values, list) and len(shap_values) == 1:
            arr = np.array(shap_values[0])  # shape may be (n_features, n_classes)
            if arr.ndim == 2 and arr.shape[0] == len(FEATURE_NAMES):
                arr = arr.T  # now (n_classes, n_features)
                for i in range(arr.shape[0]):
                    cls_label = BEHAVIOR_CLASSES[i] if i < len(BEHAVIOR_CLASSES) else str(i)
                    explanation[str(cls_label)] = {FEATURE_NAMES[j]: float(arr[i][j]) for j in range(len(FEATURE_NAMES))}
                return explanation

        # Case B: shap_values is a list with one element per class, each shaped (n_samples, n_features)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            for i, arr in enumerate(shap_values):
                arr_np = np.array(arr)
                if arr_np.ndim == 2:
                    vals = arr_np[0]
                else:
                    vals = arr_np
                cls_label = BEHAVIOR_CLASSES[i] if i < len(BEHAVIOR_CLASSES) else str(i)
                explanation[str(cls_label)] = {FEATURE_NAMES[j]: float(vals[j]) for j in range(len(FEATURE_NAMES))}
            return explanation

        # Case C: shap_values is a single 2D array shaped (n_samples, n_features)
        arr_np = np.array(shap_values)
        if arr_np.ndim == 2 and arr_np.shape[1] == len(FEATURE_NAMES):
            explanation["score"] = {FEATURE_NAMES[j]: float(arr_np[0][j]) for j in range(len(FEATURE_NAMES))}
            return explanation

    except Exception as e:
        raise RuntimeError(f"Failed to process SHAP values: {e}")

    # Fallback
    raise RuntimeError("Unrecognized SHAP output format; unable to construct explanation.")


# ======================================================
# 3. GRU regressor (reliability + deception)
# ml/gru_scores.py
# ======================================================

import tensorflow as tf

MODEL_VERSION = "gru_v2"

try:
    gru_reliability_model = tf.keras.models.load_model(BASE_DIR / "gru_model_reliability.keras")
except Exception as e:
    logger.warning("GRU reliability model could not be loaded at import time: %s", e)
    gru_reliability_model = None

try:
    gru_deception_model = tf.keras.models.load_model(BASE_DIR / "gru_model_deception.keras")
except Exception as e:
    logger.warning("GRU deception model could not be loaded at import time: %s", e)
    gru_deception_model = None


def predict_gru_scores(series):
    ts = np.array(series).reshape(1, -1, 1)

    reliability = gru_reliability_model.predict(ts, verbose=0)[0][0]
    deception = gru_deception_model.predict(ts, verbose=0)[0][0]

    return float(reliability), float(deception)


# ======================================================
# 4. Optimization layer (Pyomo TSSP)
# optimization/tssp_model.py
# ======================================================

from pyomo.environ import (
    ConcreteModel, Set, Var, Binary, NonNegativeReals,
    Objective, Constraint, SolverFactory, minimize
)

REC_COST = {
    "cooperative": 0,
    "uncertain": 20,
    "coerced": 50,
    "deceptive": 100
}

TASKS = ["Task_A", "Task_B", "Task_C"]


def solve_tssp(sources, behavior_probs, reliability, deception):

    m = ConcreteModel()
    m.S = Set(initialize=sources)
    m.T = Set(initialize=TASKS)
    m.B = Set(initialize=REC_COST.keys())

    m.x = Var(m.S, m.T, domain=Binary)
    m.y = Var(m.S, m.T, m.B, domain=NonNegativeReals)

    def stage1_cost(s):
        return 10 * (1 - reliability[s]) + 15 * deception[s]

    def objective(m):
        stage1 = sum(
            stage1_cost(s) * m.x[s, t]
            for s in m.S for t in m.T
        )

        stage2 = sum(
            behavior_probs[s][b] * REC_COST[b] * m.y[s, t, b]
            for s in m.S for t in m.T for b in m.B
        )

        reward = sum(
            5 * reliability[s] * m.x[s, t]
            for s in m.S for t in m.T
        )

        return stage1 + stage2 - reward

    m.Obj = Objective(rule=objective, sense=minimize)

    m.Assign = Constraint(
        m.S, rule=lambda m, s: sum(m.x[s, t] for t in m.T) == 1
    )

    m.Link = Constraint(
        m.S, m.T, m.B,
        rule=lambda m, s, t, b: m.y[s, t, b] <= m.x[s, t]
    )

    SolverFactory("cbc").solve(m)

    assignments = []
    for s in m.S:
        for t in m.T:
            if m.x[s, t].value > 0.5:
                expected_risk = sum(
                    behavior_probs[s][b] * REC_COST[b]
                    for b in m.B
                )
                assignments.append({
                    "source_id": s,
                    "task": t,
                    "expected_risk": round(expected_risk, 2)
                })

    return assignments


# ======================================================
# Baseline solver wrapper
# optimization/baselines.py
# ======================================================

def solve_deterministic(sources, reliability):
    return [
        {
            "source_id": s,
            "task": "Task_A",
            "expected_risk": 0.0
        }
        for s in sources
    ]


def solve_uniform(sources, reliability, deception):
    uniform_probs = {
        s: {b: 0.25 for b in REC_COST}
        for s in sources
    }
    return solve_tssp(sources, uniform_probs, reliability, deception)


# ======================================================
# EMV + EVPI Utilities
# optimization/emv.py
# ======================================================

def compute_emv(assignments):
    return round(sum(a["expected_risk"] for a in assignments), 2)


def compute_evpi(ml_emv, uniform_emv):
    return round(uniform_emv - ml_emv, 2)


# ======================================================
# PART D: FRONTEND (STREAMLIT)
# frontend/app.py
# ======================================================

import streamlit as st
import matplotlib.pyplot as plt
import shap
import requests
import pandas as pd

from api import run_optimization
from api import explain_source


# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="ML–TSSP HUMINT Tasking Dashboard",
    layout="wide"
)

st.title("ML–TSSP HUMINT Source Tasking Optimisation Dashboard")

# Informative banner if running with dummy fallback model
if MODEL_IS_DUMMY:
    st.warning(
        "XGBoost model failed to load and a dummy model is being used; SHAP explanations and realistic predictions are disabled. "
        "Place a compatible joblib XGBoost model in the 'models/' folder to enable full features."
    )


# -------------------------------------------------
# Session state
# -------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None


# -------------------------------------------------
# Source input panel
# -------------------------------------------------
st.header("Source Profiles")

sources = []

num_sources = st.slider(
    "Number of sources to simulate",
    min_value=1,
    max_value=10,
    value=3
)

for i in range(num_sources):
    st.subheader(f"Source {i + 1}")

    col1, col2 = st.columns(2)

    with col1:
        features = {
            "task_success_rate": st.slider(
                "Task Success Rate",
                0.0, 1.0, 0.6,
                key=f"tsr_{i}"
            ),
            "corroboration_score": st.slider(
                "Corroboration Score",
                0.0, 1.0, 0.5,
                key=f"cor_{i}"
            ),
            "report_timeliness": st.slider(
                "Report Timeliness",
                0.0, 1.0, 0.5,
                key=f"time_{i}"
            )
        }

    with col2:
        st.caption("GRU-predicted reliability trajectory")
        reliability_ts = [0.6, 0.65, 0.7, 0.68]
        st.line_chart(reliability_ts)

    sources.append({
        "source_id": f"SRC_{i + 1:03d}",
        "features": features,
        "reliability_series": reliability_ts
    })


# -------------------------------------------------
# Run optimisation
# -------------------------------------------------
st.divider()

if st.button("Run Optimisation"):
    payload = {
        "sources": sources,
        "seed": 42
    }

    with st.spinner("Running ML–TSSP optimisation…"):
        try:
            st.session_state.results = run_optimization(payload)
            st.success("Optimisation completed")
        except Exception as e:
            st.session_state.results = None
            st.error("Optimisation failed. Backend may be unavailable.")
            st.exception(e)

results = st.session_state.results


# -------------------------------------------------
# Results section
# -------------------------------------------------
if results is not None:
    try:
        tabs_list = st.tabs([
            "ML–TSSP",
            "Deterministic",
            "Uniform",
            "SHAP Explanations",
            "EVPI Ranking",
            "Risk vs Coverage",
            "Reliability & Deception Drift"
        ])

        if isinstance(tabs_list, (list, tuple)) and len(tabs_list) >= 7:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = tabs_list[:7]
        else:
            st.warning("Tabs could not be created in this environment; UI components will be disabled.")
            tab1 = tab2 = tab3 = tab4 = tab5 = tab6 = tab7 = None
    except Exception as e:
        st.warning("Tabs could not be created: %s" % str(e))
        tab1 = tab2 = tab3 = tab4 = tab5 = tab6 = tab7 = None

    # ---------------- ML–TSSP ----------------
    if tab1 is not None:
        with tab1:
            st.subheader("Optimised ML–TSSP Policy")

            ml_policy = results.get("policies", {}).get("ml_tssp")
            ml_emv = results.get("emv", {}).get("ml_tssp")

            if ml_policy:
                st.table(ml_policy)
            else:
                st.warning("ML–TSSP policy not available.")

            if ml_emv is not None:
                st.metric("Expected Operational Risk (EMV)", f"{ml_emv:.2f}")

    # ---------------- Deterministic ----------------
    if tab2 is not None:
        with tab2:
            st.subheader("Deterministic Assignment")
            st.caption("Ignores uncertainty and recourse")

            det_policy = results.get("policies", {}).get("deterministic")
            det_emv = results.get("emv", {}).get("deterministic")

            if det_policy:
                st.table(det_policy)
            else:
                st.warning("Deterministic policy not available.")

            if det_emv is not None:
                st.metric("Expected Operational Risk (EMV)", f"{det_emv:.2f}")

    # ---------------- Uniform ----------------
    if tab3 is not None:
        with tab3:
            st.subheader("Uniform-Probability TSSP")
            st.caption("Assumes equal likelihood of behavioural outcomes")

            uni_policy = results.get("policies", {}).get("uniform")
            uni_emv = results.get("emv", {}).get("uniform")

            if uni_policy:
                st.table(uni_policy)
            else:
                st.warning("Uniform policy not available.")

            if uni_emv is not None:
                st.metric("Expected Operational Risk (EMV)", f"{uni_emv:.2f}")

    # -------------------------------------------------
    # EVPI panel
    # -------------------------------------------------
    st.divider()
    st.header("Value of Information")

    if ml_emv is not None and uni_emv is not None:
        evpi = uni_emv - ml_emv
        st.metric("EVPI (Operational Value of ML)", f"{evpi:.2f}")
    else:
        st.warning("EVPI cannot be computed with missing EMV values.")

    with st.expander("Audit Metadata"):
        st.json(results.get("audit_log", {}))


# -------------------------------------------------
# SHAP tab UI
# -------------------------------------------------
    if tab4 is not None:
        with tab4:
            st.subheader("Source-level ML Explanations (SHAP)")

            if not sources:
                st.info("No sources available for explanation.")
            else:
                selected_source = st.selectbox(
                    "Select Source",
                    [s["source_id"] for s in sources],
                    key="shap_selected_source"
                )

                source_data = next(
                    s for s in sources if s["source_id"] == selected_source
                )

                if st.button("Explain Decision"):
                    try:
                        explanation = explain_source(source_data)
                    except Exception as e:
                        explanation = None
                        st.error("Failed to retrieve SHAP explanation.")
                        st.exception(e)

                    shap_map = explanation.get("shap_values") if explanation else None

                    if shap_map:
                        st.caption(
                            f"SHAP explanation for XGBoost behavior classifier "
                            f"(Model: {explanation.get('model', 'unknown')})"
                        )

                        behavior_keys = shap_map.keys()

                        if behavior_keys:
                            behavior = st.selectbox(
                                "Select Behavior Class",
                                list(behavior_keys),
                                key="shap_behavior"
                            )

                            shap_dict = shap_map.get(behavior, {})

                            if shap_dict:
                                fig, ax = plt.subplots()
                                ax.barh(
                                    list(shap_dict.keys()),
                                    list(shap_dict.values())
                                )
                                ax.set_title(f"Feature impact for behavior: {behavior}")
                                ax.set_xlabel("SHAP value")
                                st.pyplot(fig)
                            else:
                                st.warning("No SHAP values available for this class.")
                        else:
                            st.warning("No SHAP classes returned.")
                    else:
                        # If a helpful message was returned, show it; otherwise advise how to enable SHAP
                        msg = explanation.get("message") if explanation else None
                        if msg:
                            st.info(msg)
                        else:
                            st.info("SHAP explanation not available. Place a compatible XGBoost model (joblib) in 'models/' to enable explanations.")
# Source EVPI ranking
# -------------------------------------------------
    if tab5 is not None:
        with tab5:
            st.subheader("Source-Level EVPI Ranking")

            source_evpi = results.get("source_evpi")

            if source_evpi:
                evpi_df = (
                    pd.DataFrame(
                        source_evpi.items(),
                        columns=["Source", "EVPI"]
                    )
                    .sort_values("EVPI", ascending=False)
                )
                st.table(evpi_df)
            else:
                st.info("Source-level EVPI not available for this run.")


# -------------------------------------------------
# Risk vs Coverage
# -------------------------------------------------
    if tab6 is not None:
        with tab6:
            st.subheader("Task Coverage vs Expected Risk")

    trade = results.get("tradeoff")

    if trade:
        df = pd.DataFrame([
            {
                "Policy": k,
                "Coverage": v.get("coverage"),
                "Risk": v.get("risk")
            }
            for k, v in trade.items()
        ])

        st.scatter_chart(
            df,
            x="Risk",
            y="Coverage",
            color="Policy"
        )
    else:
        st.info("Trade-off metrics not available.")


# -------------------------------------------------
# GRU drift timeline
# -------------------------------------------------
    if tab7 is not None:
        with tab7:
            st.subheader("Reliability & Deception Drift")

        src = st.selectbox(
            "Select Source",
            [s["source_id"] for s in sources],
            key="drift_select_source"
        )

        try:
            drift = requests.get(
                f"http://backend:8000/drift/{src}",
                timeout=5
            ).json()
        except Exception:
            drift = None

        if drift:
            df = pd.DataFrame(drift)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.line_chart(
                    df.set_index("timestamp")[["reliability", "deception"]]
                )
            else:
                st.warning("Drift data malformed.")
        else:
            st.info("No drift data available for this source.")


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    "<hr style='margin-top:2rem;'>"
    "<p style='text-align:center; font-size:0.85em; color:gray;'>"
    "© 2026 ML–TSSP Research Prototype. All rights reserved."
    "</p>",
    unsafe_allow_html=True
)