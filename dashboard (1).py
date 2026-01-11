
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from tensorflow.keras.models import load_model
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition

# --- Configuration --- #
MODELS_DIR = "saved_models"
FEATURES_DIR = "saved_features"

# --- Global Definitions (from notebook context) ---
# Ensure these are consistent with what was used during training/optimization
tasks = ['Task_A', 'Task_B', 'Task_C']
behaviors = ["cooperative", "uncertain", "coerced", "deceptive"]

recourse_cost = {
    "cooperative": 0.0,
    "uncertain": 20.0,
    "coerced": 40.0,
    "deceptive": 100.0
}

total_risky_recourse_cost_sum_val = sum(recourse_cost[b] for b in ["uncertain", "coerced", "deceptive"])

# --- Model Loading --- #
@st.cache_resource
def load_ml_artifacts():
    try:
        xgb_classifier_reduced = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier_reduced.joblib"))
        gru_model_reliability = load_model(os.path.join(MODELS_DIR, "gru_model_reliability.keras"))
        gru_model_deception = load_model(os.path.join(MODELS_DIR, "gru_model_deception.keras"))
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

        with open(os.path.join(FEATURES_DIR, "xgb_classifier_features.json"), 'r') as f:
            xgb_classifier_features = json.load(f)
        with open(os.path.join(FEATURES_DIR, "regression_features.json"), 'r') as f:
            regression_features = json.load(f)

        return xgb_classifier_reduced, gru_model_reliability, gru_model_deception, le, xgb_classifier_features, regression_features
    except Exception as e:
        st.error(f"Error loading ML artifacts: {e}")
        return None, None, None, None, None, None


xgb_classifier_reduced, gru_model_reliability, gru_model_deception, le, xgb_classifier_features, regression_features = load_ml_artifacts()


# --- ML Prediction Function --- #
def get_ml_predictions_for_tssp(new_humint_data: pd.DataFrame):
    if xgb_classifier_reduced is None or gru_model_reliability is None or gru_model_deception is None or le is None:
        st.error("ML models not loaded. Cannot make predictions.")
        return {}, {}, {}

    # Ensure 'source_id' is present and set as index for easier mapping
    if 'source_id' not in new_humint_data.columns:
        raise ValueError("Input DataFrame must contain a 'source_id' column.")

    input_sources = new_humint_data['source_id'].tolist()
    new_humint_data_indexed = new_humint_data.set_index('source_id')

    # Preprocess input data for each model
    try:
        X_xgb_input = new_humint_data_indexed[xgb_classifier_features]
        X_gru_input = new_humint_data_indexed[regression_features]
    except KeyError as e:
        raise ValueError(f"Missing required feature in input data: {e}")

    # Reshape GRU-specific DataFrame
    X_gru_input_reshaped = X_gru_input.values.reshape(X_gru_input.shape[0], 1, X_gru_input.shape[1])

    # Make predictions
    behavior_class_probabilities_array = xgb_classifier_reduced.predict_proba(X_xgb_input)
    reliability_predictions_array = gru_model_reliability.predict(X_gru_input_reshaped).flatten()
    deception_predictions_array = gru_model_deception.predict(X_gru_input_reshaped).flatten()

    # Format predictions
    behavior_prob = {}
    for i, s_id in enumerate(input_sources):
        probs = behavior_class_probabilities_array[i]
        behavior_prob[s_id] = {
            le.classes_[j]: probs[j] for j in range(len(le.classes_))
        }

    reliability = {s_id: pred for s_id, pred in zip(input_sources, reliability_predictions_array)}
    deception_risk = {s_id: pred for s_id, pred in zip(input_sources, deception_predictions_array)}

    return behavior_prob, reliability, deception_risk


# --- Pyomo Optimization Function (adapted from notebook) --- #
def build_and_solve_tssp_model(
    sources: list,
    task_capacities: dict,
    behavior_prob: dict,
    reliability_scores: dict,
    deception_scores: dict, # Added for task_value calculation
    label: str = "Policy",
):
    model = ConcreteModel()

    # --- Sets ---
    model.S = Set(initialize=sources)
    model.T = Set(initialize=tasks) # Use global tasks
    model.B = Set(initialize=behaviors) # Use global behaviors

    # --- Parameters ---
    # Calculate stage1_cost based on reliability_scores
    stage1_cost = {
        (s, t): round(10 * (1 - reliability_scores.get(s, 0.5)), 2) # Default 0.5 if not found
        for s in sources for t in tasks
    }

    # Calculate task_value based on predicted scores and other features from original new_humint_data
    # This requires recreating the task_value logic for each source.
    # For simplicity, we'll assume the input `new_humint_data` (passed to Streamlit) also contains
    # the base features needed for task_value calculation, or that task_value is passed directly.
    # For now, let's just make a dummy task_value. In a real scenario, this would be computed from data.
    task_values_dict = {
        s_id: (
            0.30 * new_humint_data_input.loc[new_humint_data_input['source_id'] == s_id, 'task_success_rate'].iloc[0] +
            0.20 * new_humint_data_input.loc[new_humint_data_input['source_id'] == s_id, 'corroboration_score'].iloc[0] +
            0.20 * reliability_scores.get(s_id, 0.5) + 
            0.15 * new_humint_data_input.loc[new_humint_data_input['source_id'] == s_id, 'report_timeliness'].iloc[0] +
            0.10 * new_humint_data_input.loc[new_humint_data_input['source_id'] == s_id, 'handler_confidence'].iloc[0] -
            0.05 * deception_scores.get(s_id, 0.5)
        ).clip(0.0) # Ensure non-negative
        for s_id in sources
    }

    model.Stage1Cost = Param(model.S, model.T, initialize=stage1_cost)
    model.RecourseCost = Param(model.B, initialize=recourse_cost) # Use global recourse_cost
    model.BehaviorProb = Param(model.S, model.B, initialize=lambda m, s, b: behavior_prob.get(s, {}).get(b, 0.0))
    model.TaskCapacity = Param(model.T, initialize=task_capacities)
    model.TotalRiskyRecourseCostSum = Param(initialize=total_risky_recourse_cost_sum_val) # Use global sum

    # --- Decision variables ---
    model.x = Var(model.S, model.T, domain=Binary)
    model.y = Var(model.S, model.T, model.B, domain=NonNegativeReals)

    # --- Constraints ---
    def source_assignment_rule(m, s):
        return sum(m.x[s, t] for t in m.T) == 1
    model.SourceAssignment = Constraint(model.S, rule=source_assignment_rule)

    def task_capacity_rule(m, t):
        return sum(m.x[s, t] for s in m.S) <= m.TaskCapacity[t]
    model.TaskCap = Constraint(model.T, rule=task_capacity_rule)

    def min_task_use_rule(m, t):
        return sum(m.x[s, t] for s in m.S) >= 1
    model.MinTaskUse = Constraint(model.T, rule=min_task_use_rule)

    def recourse_link_rule(m, s, t, b):
        return m.y[s, t, b] <= m.x[s, t]
    model.RecourseLink = Constraint(model.S, model.T, model.B, rule=recourse_link_rule)

    def recourse_proportionality_rule(m, s, t, b):
        if b == "cooperative":
            return m.y[s, t, b] == 0 * m.x[s,t]
        elif b in ["uncertain", "coerced", "deceptive"]:
            return m.y[s, t, b] == (m.x[s, t] / m.TotalRiskyRecourseCostSum) * m.RecourseCost[b]
        else:
            return Constraint.Skip
    model.RecourseProportionality = Constraint(model.S, model.T, model.B, rule=recourse_proportionality_rule)

    # --- Objective ---
    def objective_rule(m):
        stage1 = sum(
            m.Stage1Cost[s, t] * m.x[s, t]
            for s in m.S for t in m.T
        )
        stage2 = sum(
            m.BehaviorProb[s, b] *
            m.RecourseCost[b] *
            m.y[s, t, b]
            for s in m.S for t in m.T for b in m.B
        )
        return stage1 + stage2

    model.Obj = Objective(rule=objective_rule, sense=minimize)

    # --- Solve ---
    solver = SolverFactory("glpk", executable="/usr/bin/glpsol")
    result = solver.solve(model, tee=False)
    model.solutions.load_from(result)

    # Helper function to calculate total value gained
    def _calculate_total_value_gained(solved_model, task_values_dict_inner):
        total_value_gained = 0
        if task_values_dict_inner is None:
            return total_value_gained

        if hasattr(solved_model, 'Obj') and solved_model.Obj.expr is not None:
            for s_id in solved_model.S:
                for t_id in solved_model.T:
                    x_val = value(solved_model.x[s_id, t_id], exception=False)
                    if x_val is not None and x_val > 0.5:
                        total_value_gained += task_values_dict_inner.get(s_id, 0)
                        break
        return total_value_gained

    # --- Metrics ---
    if result.solver.termination_condition != TerminationCondition.optimal:
        if result.solver.termination_condition == TerminationCondition.infeasible:
            st.warning(f"Policy {label} is infeasible. Returning large costs.")
            return model, {
                "Policy": label,
                "Stage1Cost": float('inf'),
                "Stage2Cost": float('inf'),
                "TotalCost": float('inf'),
                "RiskExposure": float('inf'),
                "Total Value Gained": 0.0,
                "Net Value": float('-inf')
            }
        else:
            st.error(f"{label} did not solve optimally or infeasible. Termination condition: {result.solver.termination_condition}")
            return model, {
                "Policy": label,
                "Stage1Cost": float('nan'), "Stage2Cost": float('nan'), "TotalCost": float('nan'),
                "RiskExposure": float('nan'), "Total Value Gained": float('nan'), "Net Value": float('nan')
            }

    stage1_cost_val = value(sum(
        model.Stage1Cost[s, t] * model.x[s, t]
        for s in model.S for t in model.T
    ))

    stage2_cost_val = value(sum(
        model.BehaviorProb[s, b] *
        model.RecourseCost[b] *
        model.y[s, t, b]
        for s in model.S for t in model.T for b in model.B
    ))

    risky_mass = value(sum(
        model.BehaviorProb[s, b]
        for s in model.S for b in ["coerced", "deceptive"]
    ))

    total_value_gained = _calculate_total_value_gained(model, task_values_dict)
    net_value = total_value_gained - (stage1_cost_val + stage2_cost_val)

    metrics = {
        "Policy": label,
        "Stage1Cost": stage1_cost_val,
        "Stage2Cost": stage2_cost_val,
        "TotalCost": stage1_cost_val + stage2_cost_val,
        "RiskExposure": risky_mass,
        "Total Value Gained": total_value_gained,
        "Net Value": net_value
    }

    return model, metrics


# --- Streamlit UI --- #
st.set_page_config(layout="wide")
st.title("Hybrid ML-TSSP Model for HUMINT Source Management")

st.markdown("""
This dashboard integrates Machine Learning predictions with a Two-Stage Stochastic Programming (TSSP) model to optimize HUMINT source-task assignments.

**Upload new source data (CSV) or use the example data below.**
""")

# --- Example Data --- #
example_data_str = """
source_id,task_success_rate,corroboration_score,report_timeliness,handler_confidence,deception_score,ci_flag,reliability_score,scenario_probability
NEW_SRC_001,0.984967,0.903659,0.483366,0.537656,0.706109,0,0.585377,0.437983
NEW_SRC_002,0.927617,0.595792,0.823343,0.454742,0.536490,0,0.565872,0.458523
NEW_SRC_003,0.434477,0.923674,0.703420,0.400675,0.271757,0,0.520142,0.477865
NEW_SRC_004,0.672230,0.636305,0.420867,0.772753,0.409994,0,0.491920,0.402423
NEW_SRC_005,0.835658,0.930426,0.391751,0.325556,0.348083,0,0.546877,0.519225
"""

# --- Input Data --- #
uploaded_file = st.file_uploader("Upload New HUMINT Source Data (CSV)", type=["csv"])

if uploaded_file is not None:
    new_humint_data_input = pd.read_csv(uploaded_file)
else:
    st.info("Using example data. Upload a CSV to use your own data.")
    new_humint_data_input = pd.read_csv(pd.io.common.StringIO(example_data_str))

st.subheader("1. Input HUMINT Source Data")
st.dataframe(new_humint_data_input)

if new_humint_data_input.empty:
    st.warning("Please provide input data to proceed.")
else:
    # --- ML Predictions --- #
    st.subheader("2. ML Predictions (Behavior Probabilities, Reliability, Deception)")
    try:
        behavior_probs, reliability_scores, deception_risks = get_ml_predictions_for_tssp(new_humint_data_input.copy())

        st.write("**Predicted Behavior Probabilities:**")
        st.json(behavior_probs)

        st.write("**Predicted Reliability Scores:**")
        st.json(reliability_scores)

        st.write("**Predicted Deception Risks:**")
        st.json(deception_risks)

    except Exception as e:
        st.error(f"ML Prediction Error: {e}")
        behavior_probs, reliability_scores, deception_risks = {}, {}, {}

    if behavior_probs and reliability_scores and deception_risks:
        # --- TSSP Optimization --- #
        st.subheader("3. TSSP Optimization Results")

        # Collect current task capacities (can be made dynamic in Streamlit if needed)
        current_task_capacities = {"Task_A": 35, "Task_B": 35, "Task_C": 35}

        try:
            model_tssp, metrics_tssp = build_and_solve_tssp_model(
                sources=new_humint_data_input['source_id'].tolist(),
                task_capacities=current_task_capacities,
                behavior_prob=behavior_probs,
                reliability_scores=reliability_scores,
                deception_scores=deception_risks,
                label="ML-TSSP Live"
            )

            st.write("**Optimal Task Assignments:**")
            assignments = []
            if model_tssp.Obj.expr is not None and model_tssp.Obj() != float('inf'):
                for s_id in model_tssp.S:
                    for t_id in model_tssp.T:
                        if value(model_tssp.x[s_id, t_id]) > 0.5:
                            assignments.append({"Source ID": s_id, "Assigned Task": t_id})
                            break
                st.dataframe(pd.DataFrame(assignments))

                st.write("**Performance Metrics:**")
                metrics_df = pd.DataFrame([metrics_tssp])
                st.dataframe(metrics_df)

            else:
                st.warning("TSSP Model did not solve optimally or was infeasible.")
                st.json(metrics_tssp) # Show raw metrics for debugging

        except Exception as e:
            st.error(f"TSSP Optimization Error: {e}")

