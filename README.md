# 🏦 AI Financial Copilot — Multi‑Agent Loan Risk Analyzer

## 📌 Overview
The AI Financial Copilot is a multi‑agent system that predicts loan default risk using XGBoost, interprets results with SHAP explainability, and delivers transparent business decisions through a Streamlit SaaS‑style dashboard.

This project demonstrates end‑to‑end AI engineering: model training, explainability, retrieval‑augmented generation (RAG), orchestration, MLOps tracking, and deployment‑ready UI.

---

## ✨ Key Features

### 1.Multi‑Agent Architecture

Classifier Agent → ML probability

Risk Analyzer → Debt‑to‑Income ratio

Decision Agent → Business rules (APPROVE / REJECT / REVIEW)

Explainer Agent → Human‑readable explanation

### 2.Explainable AI (XAI)

SHAP feature importance

Confusion matrix, ROC curve, metrics dashboard

### 3.Retrieval‑Augmented Generation (RAG)

FAISS + HuggingFace embeddings for contextual knowledge

### 4.MLOps Integration

MLflow experiment tracking (parameters, metrics, artifacts, predictions)

### 5.Streamlit SaaS UI

Gradient theme, interactive forms, decision banners, risk gauge, analytics dashboard

---

## 🛠️ Tech Stack

* **Languages & Frameworks:** Python, Pandas, NumPy, scikit‑learn, XGBoost

* **Explainability:** SHAP, Plotly

* **RAG:** FAISS, LangChain, Sentence‑Transformers

* **UI:** Streamlit, streamlit‑option‑menu, Plotly

* **MLOps:** MLflow, Joblib

* **LLM Integration:** Ollama (Llama3 local model)

---

## 📊 Screenshots

## Loan Predictor
<img width="1827" height="573" alt="image" src="https://github.com/user-attachments/assets/a648c0da-20e8-480a-bfaa-2a3b67489f17" />

<img width="1558" height="796" alt="image" src="https://github.com/user-attachments/assets/8ceea452-3eb1-40ee-a355-404365e30d6f" />

<img width="1881" height="731" alt="image" src="https://github.com/user-attachments/assets/ebdd56c9-adfa-4b90-8ec6-5a74e4074c28" />

---

##⚙️ Setup & Run

### Clone repo

git clone https://github.com/yourusername/ai-financial-copilot.git

cd ai-financial-copilot

### Install dependencies

pip install -r requirements.txt

### Run Streamlit app

streamlit run app.py

---

## 📈 Business Impact

✅ Transparent loan risk decisions for financial institutions

✅ Human‑readable explanations for compliance and customer trust

✅ Modular design for scalability and integration into enterprise workflows

---

## 📂 Project Structure

```
src/
  agents/          # Classifier, Risk Analyzer, Decision, Explainer
  mlops/           # MLflow tracker
  rag/             # FAISS + embeddings
  orchestrator.py  # Pipeline connector
app.py             # Streamlit UI
models/            # Trained models (ignored in .gitignore)
data/              # Processed datasets (ignored in .gitignore)
reports/           # SHAP plots, metrics
```






