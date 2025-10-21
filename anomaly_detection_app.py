import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import recall_score
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

# Top-N Flagged Transactions Table
st.subheader(f"Top {top_n} Flagged Transactions (Filtered)")
top_anomalies = filtered_df[filtered_df["Flagged"] == 1].sort_values(by="Anomaly_Score", ascending=False)
st.dataframe(top_anomalies)

st.download_button(
    label="Download Top-N Anomalies as CSV",
    data=top_anomalies.to_csv(index=False),
    file_name="top_anomalies.csv",
    mime="text/csv"
)

# Pie chart: Anomaly distribution by type
st.subheader("Anomaly Distribution by Transaction Type")
pie_data = filtered_df[filtered_df["Flagged"] == 1]["type"].value_counts().reset_index()
pie_data.columns = ["Transaction Type", "Count"]

fig_pie = px.pie(
    pie_data,
    names="Transaction Type",
    values="Count",
    title="Anomalies by Transaction Type",
    height=500
)
st.plotly_chart(fig_pie, use_container_width=True)

# Anomaly Distribution: Weekday vs Weekend
st.subheader("Anomaly Distribution: Weekday vs Weekend")
weekend_stats = filtered_df[filtered_df["Flagged"] == 1]["is_weekend"].value_counts().reset_index()
weekend_stats.columns = ["is_weekend", "Anomaly Count"]
weekend_stats["is_weekend"] = weekend_stats["is_weekend"].map({0: "Weekday", 1: "Weekend"})

fig_weekend = px.bar(
    weekend_stats,
    x="is_weekend",
    y="Anomaly Count",
    color="is_weekend",
    color_discrete_map={"Weekday": "blue", "Weekend": "red"},
    title="Anomalies by Weekday vs Weekend",
    height=500
)
st.plotly_chart(fig_weekend, use_container_width=True)

# Box Plot: Amount Distribution by Day Type
st.subheader("Amount Distribution of Anomalies: Weekday vs Weekend")
anomaly_df = filtered_df[filtered_df["Flagged"] == 1].copy()
anomaly_df["is_weekend"] = anomaly_df["is_weekend"].map({0: "Weekday", 1: "Weekend"})

fig_amount_box = px.box(
    anomaly_df,
    x="is_weekend",
    y="amount",
    color="is_weekend",
    title="Transaction Amounts of Anomalies by Day Type",
    height=500
)
st.plotly_chart(fig_amount_box, use_container_width=True)

#  Behavioral Feature Comparison
st.subheader("Anomaly Distribution: Weekday vs Weekend")
behavioral_means = anomaly_df.groupby("is_weekend")[["account_age", "is_frequent_sender", "is_frequent_receiver"]].mean().reset_index()
melted = behavioral_means.melt(id_vars="is_weekend", var_name="Feature", value_name="Average")

fig_behavior = px.bar(
    melted,
    x="Feature",
    y="Average",
    color="is_weekend",
    barmode="group",
    title="Average Behavioral Features by Day Type (Anomalies)",
    height=500
)
st.plotly_chart(fig_behavior, use_container_width=True)

# Feature importance based on mean difference
exclude_cols = ["Anomaly_Score", "Flagged", "Top_Feature", "SHAP_Value"]
feature_options = [col for col in X_test_display.columns if col not in exclude_cols]

st.subheader("Feature Importance (Top-N Anomalies vs Normal)")
feature_importance = {}
for feature in feature_options:
    if pd.api.types.is_numeric_dtype(X_test_display[feature]):
        mean_flagged = X_test_display.loc[top_indices, feature].mean()
        mean_normal = X_test_display.loc[X_test_display["Flagged"] == 0, feature].mean()
        feature_importance[feature] = abs(mean_flagged - mean_normal)

importance_df = pd.DataFrame({
    "Feature": list(feature_importance.keys()),
    "Mean Difference": list(feature_importance.values())
}).sort_values(by="Mean Difference", ascending=False)

fig_feat = px.bar(
    importance_df,
    x="Feature",
    y="Mean Difference",
    title="Top Features Differentiating Anomalies",
    height=500
)
st.plotly_chart(fig_feat, use_container_width=True)

# Fraud trend over time (only anomalies)
st.subheader("Fraud Trends Over Time")
trend_data = filtered_df[filtered_df["Flagged"] == 1].groupby("step")["Flagged"].count().reset_index()
fig_trend = px.line(
    trend_data,
    x="step",
    y="Flagged",
    markers=True,
    title="Fraud Trend Over Time",
    height=500
)
st.plotly_chart(fig_trend, use_container_width=True)

# Anomaly score distribution
sns.set_theme(style="whitegrid")
st.subheader("Anomaly Score Distribution")
fig_score = plt.figure(figsize=(7, 4))
plt.hist(scores, bins=50, color="skyblue", edgecolor="black")
plt.axvline(scores[top_indices[0]], color="red", linestyle="--", label="Top-N Cutoff")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Distribution of Anomaly Scores")
plt.legend()
st.pyplot(fig_score)

# Interactive scatter plot
st.subheader("Interactive Scatter Plot: Anomalous vs Normal")
default_x = "hour_of_the_day" if "hour_of_the_day" in feature_options else feature_options[0]
default_y = "amount" if "amount" in feature_options else feature_options[1]

feature_x = st.selectbox("Select X-axis feature", options=feature_options, index=feature_options.index(default_x))
feature_y = st.selectbox("Select Y-axis feature", options=feature_options, index=feature_options.index(default_y))

fig_scatter = px.scatter(
    filtered_df,
    x=feature_x,
    y=feature_y,
    color=filtered_df["Flagged"].map({0: "Normal", 1: "Anomalous"}),
    color_discrete_map={"Normal": "blue", "Anomalous": "red"},
    hover_data=["Anomaly_Score", "type", "step"],
    title="Anomalous vs Normal Transactions",
    height=500
)
st.plotly_chart(fig_scatter, use_container_width=True)
