# 🛰️ Telco Customer Churn — End-to-End ML (FastAPI • Docker • MLflow)
> *“In the theater of attrition, churn is treason. This system exposes it.”*

A full pipeline to analyze telecom customer behavior and **predict churn** — who stays, who leaves, and why.  
Built with **Machine Learning**, **FastAPI**, **Docker**, and **MLflow**, this project delivers data-driven insight and deployable intelligence.

---

## ✨ Highlights
- 🧠 **Models:** XGBoost, LightGBM, Random Forest, Decision Tree  
- 📊 **Experiment tracking:** MLflow logging + EDA notebooks  
- ⚡ **Real-time inference:** FastAPI REST API with Gradio UI (`/ui`)  
- 🐳 **Dockerized deployment:** production-ready, portable, fast  
- ✅ **Testing & CI:** automated with Pytest and GitHub Actions  

> *“We do not guess. We measure.”*

---

## 🗂️ Project Structure
(omitted for brevity, same as markdown provided earlier)

---

## 🧩 Problem Statement
Customer churn — the silent defection.  
This project identifies **which customers are likely to leave** a telecom provider using behavioral, demographic, and billing data.  
It transforms messy data into actionable insight and deploys predictive intelligence as a scalable API.

---

## 🧠 Models & Tracking
- **Decision Tree:** baseline interpretability  
- **Random Forest:** ensemble reliability  
- **XGBoost:** precision powerhouse  
- **LightGBM:** speed-optimized accuracy  
- **MLflow:** experiment tracking for metrics, params, and models  

> *“Every experiment is a confession written in metrics.”*

---

2️⃣ Prepare Data & Run Pipeline
```bash
python scripts/prepare_processed_data.py
python scripts/run_pipeline.py
```

Loads, validates, and cleans data

Performs feature engineering

Trains models and logs runs to MLflow

3️⃣ Launch MLflow UI
export MLFLOW_TRACKING_URI="file:./mlruns"
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI --port 5000


Visit http://127.0.0.1:5000
 to browse experiment results.

## 🕵️ Author
**Sid — Data Scientist**  
> *“Precision is my protest. Insight, my revolution.”*  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://linkedin.com/in/yourprofile)  
[![GitHub](https://img.shields.io/badge/GitHub-black)](https://github.com/yourusername)

---

## ⚖️ License
MIT License — free to use, but respect the code.  
Even spies have ethics.
