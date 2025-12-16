# 🧠 Smart Loan Approval ML Project

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.101.0-brightgreen.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Project Overview

This project implements a **supervised machine learning pipeline** to predict **loan approval** based on applicant financial and personal information.

Features:

* **EDA (Exploratory Data Analysis)**
* **Preprocessing & feature encoding**
* **Model training (Logistic Regression & Random Forest)**
* **Hyperparameter tuning & explainability**
* **API deployment using FastAPI**
* **Interactive frontend for loan prediction**

---

## Project Structure

```
smart-loan-approval/
│
├─ data/
│   └─ loan_approval_dataset.csv       # Dataset
│
├─ models/
│   └─ loan_approval_pipeline.pkl     # Saved ML pipeline
│
├─ notebooks/
│   └─ loan_approval_complete.ipynb   # Combined notebook: EDA → Training → Tuning → Save
│
├─ api/
│   └─ main.py                        # FastAPI backend
│
├─ frontend/
│   └─ index.html                     # Interactive loan approval UI
│
└─ README.md
```

---

## Dataset

| Column                   | Description                                   |
| ------------------------ | --------------------------------------------- |
| loan_id                  | Unique loan application ID                    |
| no_of_dependents         | Number of dependents                          |
| education                | Applicant education (Graduate / Not Graduate) |
| self_employed            | Self employed? (Yes / No)                     |
| income_annum             | Annual income                                 |
| loan_amount              | Requested loan amount                         |
| loan_term                | Loan term in months                           |
| cibil_score              | Credit score                                  |
| residential_assets_value | Value of residential assets                   |
| commercial_assets_value  | Value of commercial assets                    |
| luxury_assets_value      | Value of luxury assets                        |
| bank_asset_value         | Value of bank assets                          |
| loan_status              | Target variable (Approved / Rejected)         |

---

## Installation

1. **Clone the repository**

```bash
git clone <repo_url>
cd smart-loan-approval
```

2. **Create a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

* Open `notebooks/loan_approval_complete.ipynb`
* Steps:

1. EDA & visualizations
2. Preprocessing (numeric & categorical pipelines)
3. Train/test split
4. Train Logistic Regression & Random Forest
5. Hyperparameter tuning with GridSearchCV
6. Feature importance & threshold optimization
7. Save final pipeline (`loan_approval_pipeline.pkl`)

---

## FastAPI Backend

1. Navigate to API folder:

```bash
cd api
```

2. Start server:

```bash
uvicorn main:app --reload
```

* Server: `http://127.0.0.1:8000`
* Endpoint: `POST /predict`

**Payload example:**

```json
{
  "income_annum": 500000,
  "loan_amount": 200000,
  "cibil_score": 750,
  "education": "Graduate",
  "self_employed": "No"
}
```

**Response:**

```json
{
  "approved": true,
  "approval_probability": 0.87
}
```

---

## Frontend

* Open `frontend/index.html` in browser
* Enter applicant details
* Click **Check Approval**
* Displays **Approved ✅** or **Rejected ❌** with probability
* Provides hints for categorical inputs (dropdowns for `education` and `self_employed`)

---

## Model Evaluation

Metrics:

* **Accuracy**, **F1-Score**, **ROC-AUC**, **Confusion Matrix**
* Default decision threshold: **0.4**
* Feature importance & coefficients for explainability

---

## Example Applications

**Approved:**

```json
{
  "income_annum": 600000,
  "loan_amount": 200000,
  "cibil_score": 750,
  "education": "Graduate",
  "self_employed": "No"
}
```

**Rejected:**

```json
{
  "income_annum": 200000,
  "loan_amount": 400000,
  "cibil_score": 500,
  "education": "Not Graduate",
  "self_employed": "Yes"
}
```

---

## References

* [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
* [FastAPI Documentation](https://fastapi.tiangolo.com/)
* [EDA & ML Best Practices](https://towardsdatascience.com/)
