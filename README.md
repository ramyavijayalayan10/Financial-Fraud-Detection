# 💳 Fraud Detection Dashboard using Isolation Forest

This project is a Streamlit-powered dashboard for detecting fraudulent financial transactions using an Isolation Forest model. It highlights the top-N most suspicious transactions based on anomaly scores and provides interactive tools for investigation, filtering, and feature attribution.

🔗 **Live App**: [Streamlit Dashboard](https://financial-fraud-detection-x9srwkbaz4uewvzzes7og7.streamlit.app/)


---

## Tech Stack
This project involved a mix of big data processing, modeling, and app development tools:


| Layer | Tools Used |
|-------|------------|
| Data Processing (6M+ rows) | **PySpark**, **Spark SQL**, **Pandas**, **NumPy** |
| Modeling | **PyOD**, **Isolation Forest**, **KNN**, **LOF** |
| Benchmarking | **scikit-learn**, **GridSearchCV**, **SHAP** |
| Development | **Google Colab** (data prep & modeling), **VS Code** (Streamlit app) |
| Deployment | **Streamlit Community Cloud** |

---
##  Modeling Journey & Why Isolation Forest Was Chosen 

Initially, I trained multiple anomaly detection models — Isolation Forest, KNN, and LOF — using PyOD. KNN and LOF were trained on a downsampled 1:10 fraud ratio dataset due to their sensitivity to class imbalance. Isolation Forest, however, was trained on the full 6M+ dataset with a 1:100 fraud ratio.

After benchmarking all three models using precision, recall, F1 score, and ROC/PR curves, it became clear that Isolation Forest was the best fit. Even though its precision was low, it consistently delivered high recall — which is more important in fraud detection. Missing a fraud is costlier than flagging a few false positives.

I used a semi-supervised approach for tuning: y_train was used only for scoring during GridSearchCV, not for training. This helped select hyperparameters that improved recall without compromising the unsupervised nature of the model.


---

## Precision vs Recall Tradeoff

Fraud detection favors **high recall** — we want to catch as many fraudulent cases as possible, even if some false positives occur.  
While our precision is low, the **high recall ensures fewer frauds go undetected**, which is critical in real-world scenarios where missing a fraud is costlier than investigating a false alarm.

---

##  What the App Offers 
This app acts as a dashboard that is designed to mimic how fraud analysts work — reviewing a ranked list of suspicious transactions rather than relying on arbitrary thresholds.

### 🔍 Top-N Anomaly Detection
- Flags the top-N transactions with the highest anomaly scores.
- Avoids arbitrary thresholding and aligns better with real-world workflows where analysts review the riskiest cases first.

### 📈 Interactive Visualizations
- **Scatter plots** to compare anomalous vs normal transactions.
- **Precision-Recall and ROC curves** to evaluate model performance.
- **Fraud trend over time** to spot spikes and patterns.

### 🧠 SHAP Feature Attribution
- **Global feature importance** via SHAP summary plots.
- **Local explanations** for individual transactions using SHAP force plots.

### 🧾 Filters and Export
- Filter transactions by type and time window.
- Download flagged transactions for offline review.

---
## How Fraud Is Flagged ?
Once the model scores the test data, the top-N transactions are flagged based on their anomaly scores. These are visualized in the dashboard and enriched with SHAP insights to explain why they were flagged. Analysts can filter by transaction type or time, explore patterns, and export the results.

--- 
## How to Use the App ?
1. Use the **Top-N slider** to select how many transactions to flag.
2. Apply filters to focus on specific transaction types or time periods.
3. Review flagged transactions in the anomaly table.
4. Use SHAP insights to understand why a transaction was flagged.
5. Export the results for further investigation.

---

## 📁 Project Structure
```bash
fraud_dashboard/
├── anomaly_detection_app.py                      # Streamlit app
├── models/
│   └── iforest_model.pkl       # Trained Isolation Forest model
├── data/
│   ├── X_test_scaled.npy
│   ├── y_test.npy
│   └── X_test_display.csv
├── notebook_scripts/
    ├── Anomaly_detection_in_Financial_Transactions.pynb
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── LICENSE.md

```
---

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run anomaly_detection_app.py
```
---
## 🔐 License


This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.  
View full terms in [LICENSE.md](LICENSE.md) or [here](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---
