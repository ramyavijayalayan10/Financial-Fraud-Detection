import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_shap as st_shap
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve, auc, recall_score
from pyod.models.iforest import IForest

# Custom recall scorer used during model training
def anomaly_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

# Load model and data
model = joblib.load("models/iforest_model.pkl")
X_train_scaled = np.load("data/X_train_scaled.npy")
X_test_scaled = np.load("data/X_test_scaled.npy")
y_test = np.load("data/y_test.npy")
X_test_display = pd.read_csv("data/X_test_display.csv")

# Sidebar: Top-N selector
st.sidebar.title("Fraud Detection Settings")
top_n = st.sidebar.slider("Number of Top Anomalies to Flag", min_value=10, max_value=200, value=100)

# Compute anomaly scores
scores = model.decision_function(X_test_scaled)
top_indices = np.argsort(scores)[-top_n:]
adjusted_preds = np.zeros_like(scores, dtype=int)
adjusted_preds[top_indices] = 1

# Add predictions to display DataFrame
X_test_display["Anomaly_Score"] = scores
X_test_display["Flagged"] = adjusted_preds

# Sidebar: Filters
st.sidebar.subheader("Filter Transactions")
type_options = X_test_display["type"].unique()
selected_types = st.sidebar.multiselect("Select transaction types", options=type_options, default=list(type_options))
min_step, max_step = int(X_test_display["step"].min()), int(X_test_display["step"].max())
selected_range = st.sidebar.slider("Select time range (step)", min_value=min_step, max_value=max_step, value=(min_step, max_step))

# Apply filters
filtered_df = X_test_display[
    (X_test_display["type"].isin(selected_types)) &
    (X_test_display["step"].between(selected_range[0], selected_range[1]))
]

# Main dashboard
st.title("💳 Financial Fraud Detection Dashboard")
st.subheader(f"Top {top_n} Flagged Transactions (Filtered)")
top_anomalies = filtered_df[filtered_df["Flagged"] == 1].sort_values(by="Anomaly_Score", ascending=False)
st.dataframe(top_anomalies)

st.download_button(
    label="Download Top-N Anomalies as CSV",
    data=top_anomalies.to_csv(index=False),
    file_name="top_anomalies.csv",
    mime="text/csv"
)

# Anomaly score distribution
st.subheader("Anomaly Score Distribution")
fig_score = plt.figure()
plt.hist(scores, bins=50, color="skyblue", edgecolor="black")
plt.axvline(scores[top_indices[0]], color="red", linestyle="--", label="Top-N Cutoff")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Distribution of Anomaly Scores")
plt.legend()
st.pyplot(fig_score)

# Scatter plot
st.subheader("Scatter Plot: Anomalous vs Non-Anomalous")
exclude_cols = ["Anomaly_Score", "Flagged", "Top_Feature", "SHAP_Value"]
feature_options = [col for col in X_test_display.columns if col not in exclude_cols]
feature_x = st.selectbox("Select X-axis feature", options=feature_options)
feature_y = st.selectbox("Select Y-axis feature", options=feature_options)

fig_scatter, ax = plt.subplots()
colors = filtered_df["Flagged"].map({0: "gray", 1: "red"})
ax.scatter(filtered_df[feature_x], filtered_df[feature_y], c=colors, alpha=0.6)
ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)
ax.set_title("Anomalous vs Non-Anomalous Transactions")
st.pyplot(fig_scatter)

# Fraud trend over time
st.subheader("Fraud Trends Over Time")
trend_data = filtered_df.groupby("step")["Flagged"].sum().reset_index()
fig_trend, ax = plt.subplots()
ax.plot(trend_data["step"], trend_data["Flagged"], marker="o", color="crimson")
ax.set_xlabel("Time Step")
ax.set_ylabel("Number of Flagged Transactions")
ax.set_title("Fraud Trend Over Time")
st.pyplot(fig_trend)

# Precision-Recall and ROC Curve
precision, recall, _ = precision_recall_curve(y_test, scores)
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

st.subheader("Precision-Recall Curve")
fig_pr = plt.figure()
plt.plot(recall, precision, label="Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
st.pyplot(fig_pr)

st.subheader("ROC Curve")
fig_roc = plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot(fig_roc)

# SHAP explanations
if st.checkbox("Show SHAP explanations"):
    st.subheader("SHAP Feature Importance")

    # Extract raw IsolationForest from PyOD + GridSearchCV
    raw_iforest = model.best_estimator_.clf

    @st.cache_data
    def compute_shap_values():
        # Use SHAP TreeExplainer for fast computation
        explainer = shap.TreeExplainer(raw_iforest)
        shap_values = explainer.shap_values(X_test_scaled[top_indices])
        return explainer, shap_values

    with st.spinner("Computing SHAP values..."):
        explainer, shap_values = compute_shap_values()

    st.success("SHAP values computed!")

    # SHAP summary bar plot
    fig_summary = plt.figure()
    shap.summary_plot(shap_values, X_test_scaled[top_indices], plot_type="bar", show=False)
    st.pyplot(fig_summary)

    # SHAP force plot for individual transaction
    st.subheader("SHAP Force Plot for Individual Transaction")
    row_index = st.selectbox("Select transaction index", options=top_indices)

    try:
        idx_local = np.where(top_indices == row_index)[0][0]
        st_shap.force_plot(explainer.expected_value, shap_values[idx_local], X_test_scaled[row_index])
    except Exception as e:
        st.error(f"Could not generate force plot: {e}")

    # Extract top contributing feature for each flagged anomaly
    top_features = [X_test_display.columns[np.argmax(np.abs(shap_values[i]))] for i in range(len(top_indices))]
    top_feature_values = [shap_values[i][np.argmax(np.abs(shap_values[i]))] for i in range(len(top_indices))]

    # Add SHAP insights to display DataFrame
    X_test_display.loc[top_indices, "Top_Feature"] = top_features
    X_test_display.loc[top_indices, "SHAP_Value"] = top_feature_values

    # Display enriched anomaly table
    st.subheader("Top-N Anomalies with SHAP Insights")
    st.dataframe(X_test_display.loc[top_indices, ["Flagged", "Anomaly_Score", "Top_Feature", "SHAP_Value"] + feature_options])
