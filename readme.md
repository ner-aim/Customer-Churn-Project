# 🛰️  Customer Churn - End-to-End ML (FastAPI • Docker • MLflow)
> *“In the theater of attrition, churn is treason. This system exposes it.”*

A full pipeline to analyze telecom customer behavior and **predict churn** - who stays, who leaves, and why.  
Built with **Machine Learning**, **FastAPI**, **Docker**, and **MLflow**, this project delivers data-driven insight and deployable intelligence.

---

## 🛠️ Technical Architecture

### Machine Learning & Data Science
- **Algorithms**: XGBoost, LightGBM, Random Forest, Decision Tree
- **Optimization**: Optuna (30 trials, recall-focused)
- **Experiment Tracking**: MLflow
- **Explainability**: SHAP values
- **Data Validation**: Great Expectations

### Backend & Deployment
- **API**: FastAPI (async endpoints)
- **Containerization**: Docker (multi-stage builds)
- **Orchestration**: AWS ECS/Fargate
- **CI/CD**: GitHub Actions (automated Docker builds → ECR)
- **Infrastructure**: VPC, ALB, Security Groups

### Frontend
- **UI Framework**: Gradio (interactive prediction interface)

### Development
- **Environment**: Jupyter, Python 3.11
- **Testing**: Pytest
- **Version Control**: Git/GitHub

> *“We do not guess. We measure.”*

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────────┐
│                    CUSTOMER CHURN PREDICTION SYSTEM                 │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│ Preprocessing │────▶│ Model Training│
│  (CSV)       │     │ & Validation  │     │ (XGBoost)     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                   │
                     ┌──────────────┐             │
                     │ Optuna Tuning│◀────────────┘
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │ MLflow       │
                     │ Experiment   │
                     │ Tracking     │
                     └──────┬───────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                    ┌────────▼────────┐
│  FastAPI       │                    │  Gradio UI      │
│  REST API      │                    │  Web Interface  │
└───────┬────────┘                    └─────────────────┘
        │
┌───────▼────────┐
│  Docker        │
│  Container     │
└───────┬────────┘
        │
┌───────▼────────────────────────────────────────────────┐
│              AWS CLOUD INFRASTRUCTURE                  │
│                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ GitHub   │───▶│  ECR     │───▶│   ECS    │       │
│  │ Actions  │    │ Registry │    │ Fargate  │       │
│  └──────────┘    └──────────┘    └────┬─────┘       │
│                                        │              │
│                                  ┌─────▼─────┐       │
│                                  │    ALB    │       │
│                                  │ (Port 80) │       │
│                                  └─────┬─────┘       │
│                                        │              │
└────────────────────────────────────────┼──────────────┘
                                         │
                                    ┌────▼────┐
                                    │  Users  │
                                    │ 🌐 API  │
                                    └─────────┘

Key Metrics: 92.5% Recall | <10ms Latency | Dockerized | CI/CD Automated


```
---

## 📁 Project Structure
```
Customer-Churn-Project/
│
├── notebooks/
│   ├── EDA.ipynb              # Exploratory analysis + SHAP
│   └── modeling.ipynb         # Model development
│
├── src/
│   ├── app/
│   │   ├── main.py           # FastAPI endpoints
│   │   └── app.py            # Gradio UI
│   ├── data/
│   │   ├── load_data.py      # Data ingestion
│   │   └── preprocess.py     # Feature engineering
│   ├── models/
│   │   ├── train.py          # Training logic
│   │   ├── tune.py           # Optuna optimization
│   │   └── evaluate.py       # Metrics calculation
│   └── serving/
│       └── inference.py      # Prediction pipeline
│
├── scripts/
│   ├── run_pipeline.py       # End-to-end training
│   └── test_*.py             # Unit tests
│
├── mlruns/                    # MLflow artifacts
├── .github/workflows/         # CI/CD configuration
├── dockerfile                 # Container definition
└── requirements.txt           # Dependencies
```

---

## 🧩 Problem Statement
Customer churn: the silent defection.  
This project identifies **which customers are likely to leave** a telecom provider using behavioral, demographic, and billing data.  
It transforms messy data into actionable insight and deploys predictive intelligence as a scalable API.

## 🔧 Setup (Local)

### Prerequisites

- Python ≥ 3.11
  
- Docker (optional for containerized deployment)
  
- pip, virtualenv, or conda  

### 1️⃣ Clone & Install

```bash
git clone https://github.com/{username}/Customer-Churn-Project.git
cd Customer-Churn-Project-main
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
docker build -t churn-api -f dockerfile .
```

Run Container
```
docker run -d -p 8000:8000 churn-api
```



## 🏆 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **XGBoost (Tuned)** | **67.1%** | **44.3%** | **92.5%** | **59.9%** | **1.8s** |
| LightGBM | 73.3% | 49.8% | 81.8% | 61.9% | 5.7s |
| Random Forest | 75.8% | 53.2% | 72.5% | 61.4% | - |
| XGBoost (Default) | 72.2% | 48.6% | 81.0% | 60.8% | 2.4s |

*Optimized for recall to minimize missed churn cases (false negatives)*

## 📊 Model Performance

### Production Model (Tuned XGBoost)
- **Recall**: 92.2% (prioritizing churn detection)
- **Precision**: 44.3%
- **F1-Score**: 59.9%
- **ROC-AUC**: [Add score from notebook]
- **Training Time**: 1.83s
- **Inference Time**: 8.4ms per prediction

### Business Impact
- Identifies 92% of at-risk customers
- Enables proactive retention campaigns
- Optimized threshold (0.30) balances false positives vs. missed churners

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

## 💡 Key Challenges & Solutions

### 1. Class Imbalance (73/27 split)
**Solution**: Implemented class weighting + threshold optimization (0.30) to prioritize recall

### 2. Multicollinearity
**Solution**: VIF analysis + feature engineering (consolidated "No internet service" flags)

### 3. Model Interpretability
**Solution**: SHAP values showing tenure, contract type, and monthly charges as top drivers

### 4. Production Reliability
**Solution**: Great Expectations for data validation + comprehensive unit tests


## 🕵️ Author
**Sid - Data Scientist**  
> *“Precision is my protest. Insight, my revolution.”*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![AWS](https://img.shields.io/badge/AWS-ECS-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-red)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://linkedin.com/in/pottapatri)  
[![GitHub](https://img.shields.io/badge/GitHub-yellow)](https://github.com/ner-aim)

---

## ⚖️ License
MIT License. free to use, but respect the code.  
Even spies have ethics.
