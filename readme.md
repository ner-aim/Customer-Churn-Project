# 🛰️  Customer Churn - End-to-End ML (FastAPI • Docker • MLflow)
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
Telco-Customer-Churn-ML-main/
├── README.md

├── dockerfile

├── requirements.txt

├── .github/workflows/ci.yml


├── notebooks/

│ └── EDA.ipynb

├── scripts/

│ ├── prepare_processed_data.py

│ ├── run_pipeline.py

│ ├── test_fastapi.py

│ ├── test_pipeline_phase1_data_features.py

│ └── test_pipeline_phase2_modeling.py

│
└── src/

├── app/

│ ├── app.py

│ └── main.py ← FastAPI + Gradio mounted at /ui

├── data/

│ ├── load_data.py

│ └── preprocess.py

├── features/

│ └── build_features.py

├── models/

│ ├── train.py ← MLflow logging

│ ├── tune.py

│ └── evaluate.py

├── serving/

│ ├── inference.py ← Loads MLflow-exported model + schema

│ └── model/ ← MLflow artifacts baked into Docker

└── utils/

├── utils.py

└── validate_data.py

---

## 🧩 Problem Statement
Customer churn — the silent defection.  
This project identifies **which customers are likely to leave** a telecom provider using behavioral, demographic, and billing data.  
It transforms messy data into actionable insight and deploys predictive intelligence as a scalable API.

## 🔧 Setup (Local)

### Prerequisites

- Python ≥ 3.11
  
- Docker (optional for containerized deployment)
  
- pip, virtualenv, or conda  

### 1️⃣ Clone & Install

```bash
git clone https://github.com/yourusername/Telco-Customer-Churn-ML.git
cd Telco-Customer-Churn-ML-main
pip install -r requirements.txt
```
---

### 🧠 Models & Tracking
- **Decision Tree:** baseline interpretability  
- **Random Forest:** ensemble reliability  
- **XGBoost:** precision powerhouse  
- **LightGBM:** speed-optimized accuracy  
- **MLflow:** experiment tracking for metrics, params, and models  

> *“Every experiment is a confession written in metrics.”*

---

### 2️⃣ Prepare Data & Run Pipeline
```bash
python scripts/prepare_processed_data.py
python scripts/run_pipeline.py
```

-Loads, validates, and cleans data
-Performs feature engineering
-Trains models and logs runs to MLflow

### 3️⃣ Launch MLflow UI
```
export MLFLOW_TRACKING_URI="127.0.0.1:5000"
mlflow ui --127.0.0.1 --port 5000
```

Visit http://127.0.0.1:5000
 to browse experiment results.

### 🚀 Serve the Model Locally
Run FastAPI + Gradio
```
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

- Gradio UI → http://127.0.0.1:8000/ui

### 🐳 Docker Deployment
Build Image

```
docker build -t telco-churn-api -f dockerfile .
```

Run Container
```
docker run -d -p 8000:8000 telco-churn-api
```



### 📈 Model Metrics (Illustrative)

| Model         | Accuracy | ROC-AUC | Recall | Notes             |
| ------------- | -------- | ------- | ------ | ----------------- |
| Decision Tree | 0.74     | 0.78    | Medium | Baseline          |
| Random Forest | 0.80     | 0.85    | High   | Balanced          |
| XGBoost       | 0.82     | 0.87    | High   | Strong performer  |
| LightGBM      | 0.83     | 0.88    | High   | Fast and accurate |

#### 🧮 Key Insights

- Senior Citizens and month-to-month contracts are major churn drivers.
- Electronic check payments correlate strongly with churn.
- Tenure and multi-line services improve retention.
- Long-term contracts = long-term loyalty.

_“Patterns reveal themselves only to those patient enough to compute them.”_

### 🧰 Tech Stack
| Layer               | Tools                                      |
| ------------------- | ------------------------------------------ |
| Data Analysis       | Python, Pandas, NumPy, Seaborn, Matplotlib |
| Machine Learning    | Scikit-learn, XGBoost, LightGBM            |
| Experiment Tracking | MLflow                                     |
| API Layer           | FastAPI, Gradio                            |
| Deployment          | Docker                                     |
| Environment         | Jupyter Notebook, Uvicorn                  |
| Testing             | Pytest, GitHub Actions                     |


### 🧩 Design Highlights

- Training/Serving Consistency: serving layer loads feature schema from training (feature_columns.txt)

- Unified Interface: FastAPI backend and Gradio front-end share the same inference function

- MLflow Integration: every run tracked with params, metrics, and artifacts

- Containerized Deployment: portable, reproducible environment baked with model artifacts

### 🛣️ Roadmap

- 🔍 Add SHAP/LIME explainability endpoints (/explain)

- 📈 Deploy Streamlit dashboard for churn visualization

- ☁️ Cloud deployment via AWS ECS / Azure App Service

- ⚙️ Bayesian optimization using Optuna

- 🧾 Batch inference job with Parquet input/output


## 🕵️ Author
**Sid - Data Scientist**  
> *“Precision is my protest. Insight, my revolution.”*  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://linkedin.com/in/yourprofile)  
[![GitHub](https://img.shields.io/badge/GitHub-black)](https://github.com/yourusername)

---

## ⚖️ License
MIT License. free to use, but respect the code.  
Even spies have ethics.
