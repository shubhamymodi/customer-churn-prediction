# Customer Churn Prediction App

## 📌 Project Overview

Customer churn is a critical problem for subscription-based businesses. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project builds a **Machine Learning system to predict customer churn probability** using customer demographic, service usage, and billing data.

The solution includes:

- Exploratory Data Analysis (EDA)
- Feature engineering & preprocessing
- Model training and comparison
- Performance evaluation
- Model deployment with Streamlit

Users can input customer information and receive a **real-time churn probability prediction**.

---

## 🚀 Live Demo

Streamlit App:  https://customer-churn-prediction-lr-rf.streamlit.app/

---
## 📊 Dataset

This project uses the **Telco Customer Churn Dataset** available on Kaggle.

Dataset Link:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains **7043 customer records with 21 features**, including:

- Customer demographics (gender, partner, dependents, senior citizen)
- Account information (tenure, contract type, billing method)
- Services subscribed (internet, streaming, security, tech support)
- Billing information (monthly charges, total charges)
- Target variable **Churn** indicating whether the customer left the service

Each row represents a customer and each column describes attributes related to their telecom service usage and account details. The goal is to analyze this data and build machine learning models to **predict whether a customer is likely to churn**, helping businesses design targeted retention strategies.

---
## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## ⚙️ Machine Learning Pipeline

### Numerical Features

- Tenure
- MonthlyCharges
- TotalCharges

Processing steps:

- Missing value imputation
- Standard scaling

### Categorical Features

- Gender
- Contract type
- Internet service
- Payment method
- Service-related features

Processing steps:

- Missing value imputation
- One-hot encoding

The preprocessing pipeline is implemented using **Scikit-learn ColumnTransformer**.

---

## 🤖 Models Used

### Logistic Regression

- Baseline interpretable model
- Higher recall for identifying churn customers

### Random Forest

- Ensemble learning model
- Captures non-linear relationships
- Slightly stronger balanced performance

---

## 📈 Model Performance

| Model | ROC-AUC | F1 Score | Precision | Recall |
|------|------|------|------|------|
| Logistic Regression | 0.86 | 0.64 | 0.52 | 0.84 |
| Random Forest | 0.85 | 0.65 | 0.56 | 0.78 |

---

## 🔍 Key Insights

### Contract Type
Customers on **month-to-month contracts** have the highest churn probability.

### Customer Tenure
Customers with shorter tenure are significantly more likely to churn.

### Monthly Charges
Higher monthly charges increase churn likelihood.

### Internet Service Type
Customers with **fiber optic internet service** show higher churn rates.

### Lack of Value-added Services
Customers without services like:

- Online Security
- Tech Support
- Device Protection

are more likely to leave.

---

## 💡 Business Recommendations

Based on the model insights:

- Encourage **long-term contracts** through discounts or loyalty programs
- Offer **bundled services (security, tech support)** to increase retention
- Provide **retention offers to high-charge customers**
- Focus retention strategies on **new customers with low tenure**

