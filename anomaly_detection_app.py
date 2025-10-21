import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
st.metric("Total Transactions (Filtered)", len(filtered_df))
st.metric("Flagged Anomalies", filtered_df["Flagged"].sum())

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

# Feature importance based on mean difference
exclude_cols = ["Anomaly_Score", "Flagged", "Top_Feature", "SHAP_Value"]
feature_options = [col for col in X_test_display.columns if col not in exclude_cols]

st.subheader("Feature Importance (Top-N Anomalies vs Normal)")
feature_importance = {}
for feature in feature_options:
    mean_flagged = X_test_display.loc[top_indices, feature].mean()
    mean_normal = X_test_display.loc[X_test_display["Flagged"] == 0, feature].mean()
    feature_importance[feature] = abs(mean_flagged - mean_normal)

importance_df = pd.DataFrame({
    "Feature": list(feature_importance.keys()),
    "Mean Difference": list(feature_importance.values())
}).sort_values(by="Mean Difference", ascending=False)

fig_feat = px.bar(importance_df, x="Feature", y="Mean Difference", title="Top Features Differentiating Anomalies")
st.plotly_chart(fig_feat, use_container_width=True)

# Interactive scatter plot
st.subheader("Interactive Scatter Plot: Anomalous vs Non-Anomalous")
feature_x = st.selectbox("Select X-axis feature", options=feature_options)
feature_y = st.selectbox("Select Y-axis feature", options=feature_options)

fig_scatter = px.scatter(
    filtered_df,
    x=feature_x,
    y=feature_y,
    color=filtered_df["Flagged"].map({0: "Normal", 1: "Anomalous"}),
    hover_data=["Anomaly_Score", "type", "step"],
    title="Anomalous vs Non-Anomalous Transactions"
)
st.plotly_chart(fig_scatter, use_container_width=True)

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
