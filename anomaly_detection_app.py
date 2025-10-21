import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_shap as st_shap
from sklearn.metrics import precision_recall_curve, roc_curve, auc, recall_score
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF


def anomaly_recall(y_true, y_pred):
    return recall_score(y_test, y_pred)
    

# Load model and data
model = joblib.load("models/iforest_model.pkl")
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

# Transaction type filter
type_options = X_test_display["type"].unique()
selected_types = st.sidebar.multiselect("Select transaction types", options=type_options, default=list(type_options))

# Time filter (assuming 'step' is time in hours)
min_step, max_step = int(X_test_display["step"].min()), int(X_test_display["step"].max())
selected_range = st.sidebar.slider("Select time range (step)", min_value=min_step, max_value=max_step, value=(min_step, max_step))

# Apply filters
filtered_df = X_test_display[
    (X_test_display["type"].isin(selected_types)) &
    (X_test_display["step"].between(selected_range[0], selected_range[1]))
]

# Show top flagged transactions
st.title("Isolation Forest Fraud Detection Dashboard")
st.subheader(f"Top {top_n} Flagged Transactions (Filtered)")
top_anomalies = filtered_df[filtered_df["Flagged"] == 1].sort_values(by="Anomaly_Score", ascending=False)
st.dataframe(top_anomalies)

# Download button
st.download_button(
    label="Download Top-N Anomalies as CSV",
    data=top_anomalies.to_csv(index=False),
    file_name="top_anomalies.csv",
    mime="text/csv"
)

# Sample background data for KernelExplainer
background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]

# Create SHAP explainer
explainer = shap.KernelExplainer(model.predict_proba, background)

# Compute SHAP values for test data
shap_values = explainer.shap_values(X_test_scaled)

if st.checkbox("Show SHAP explanations"):
    st.write("Generating SHAP values...")
    shap_values = explainer.shap_values(X_test_scaled)
    st.write("SHAP values computed!")

    # Visualize
    st.pyplot(shap.summary_plot(shap_values, X_test_scaled, show=False))


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

# Scatter plot: Anomalous vs Non-Anomalous
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

# SHAP Feature Importance
st.subheader("SHAP Feature Importance")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# SHAP summary plot
fig_summary = plt.figure()
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
st.pyplot(fig_summary)

# SHAP force plot for selected transaction
st.subheader("SHAP Force Plot for Individual Transaction")
row_index = st.selectbox("Select transaction index", options=top_indices)
st_shap.force_plot(explainer.expected_value, shap_values[row_index], X_test_scaled[row_index])

# Add SHAP insights to table
top_features = [X_test_display.columns[np.argmax(np.abs(shap_values[i]))] for i in top_indices]
top_feature_values = [shap_values[i][np.argmax(np.abs(shap_values[i]))] for i in top_indices]

X_test_display.loc[top_indices, "Top_Feature"] = top_features
X_test_display.loc[top_indices, "SHAP_Value"] = top_feature_values

st.subheader("Top-N Anomalies with SHAP Insights")
st.dataframe(X_test_display.loc[top_indices, ["Flagged", "Anomaly_Score", "Top_Feature", "SHAP_Value"] + feature_options])
