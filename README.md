# DiabetesAI — Diabetes Risk Prediction Web Application

A full-stack, machine learning-powered web application that predicts diabetes risk based on clinical health parameters. Built with Django and LightGBM, featuring a premium UI designed to match professional SaaS standards.

[![Live Website](https://img.shields.io/badge/Live%20Demo-Visit%20Site-0D9488?style=for-the-badge)](https://raghavchandna.pythonanywhere.com/)

---

## Overview

Diabetes Predictor (DiabetesAI) is an end-to-end diabetes risk prediction tool that takes 8 clinical parameters as input and returns an instant risk assessment powered by a gradient-boosted machine learning model. The application includes user authentication, prediction history tracking, a health dashboard with charts, and a fully responsive mobile interface.

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 90% |
| Recall (Diabetic class) | 91% |
| F1 Score (Diabetic class) | 87% |
| Precision (Diabetic class) | 83% |
| Optimal Threshold | 0.16 |

The threshold of 0.16 was determined using **Youden's J statistic** on the ROC curve, optimising for high recall — minimising missed diabetic cases which is the clinically safer approach.

**Dataset:** Pima Indian Diabetes Dataset  
- Total records: 768  
- Training set: 606 rows (80%)  
- Test set: 154 rows (20%)  
- Features: 8 clinical parameters  

---

## Features

- **User Authentication** — Register, login, guest mode
- **Diabetes Risk Predictor** — 8-parameter clinical input form with real-time field validation and normal range hints
- **Instant Prediction** — LightGBM model loaded at server startup for sub-second predictions
- **Health Dashboard** — Prediction history table, donut chart, sparklines, live date/time in user's local timezone
- **Prediction History** — Filterable table with date, time, result badge and risk bar
- **Responsive Design** — Full mobile support with bottom navigation bar on small screens
---

## Technologies Used

### Frontend
- **HTML5** — Semantic page structure
- **CSS3** — Fully custom dark theme (no frameworks or Bootstrap), CSS variables, grid/flexbox layouts, keyframe animations
- **Vanilla JavaScript** — Form validation, local timezone conversion, dynamic UI interactions
- **Chart.js** — Donut chart and sparkline mini-charts on the dashboard
- **Google Fonts** — Outfit (UI) + DM Serif Display (headings)

### Backend
- **Python** — Core language
- **Django 4.0.6** — Web framework, routing, authentication, ORM
- **SQLite** — Database for users and prediction history

### Machine Learning
- **LightGBM** — Gradient boosted decision tree classifier
- **Scikit-learn** — StandardScaler for feature scaling, ROC curve for threshold optimisation
- **Pandas** — Data manipulation and preprocessing
- **NumPy** — Numerical operations and threshold application

---

## Project Structure
```
Diabetes Predictor/
├── diabetes_predictor_website/   # Django project settings
├── predictor/                    # Main Django app
│   ├── ml_model/
│   │   ├── best_lgb_model.pkl   # Trained LightGBM model
│   │   └── scaler.pkl           # Fitted StandardScaler
│   ├── templates/predictor/
│   │   ├── login.html           # Login page
│   │   ├── register.html        # Registration page
│   │   ├── predict.html         # Prediction form
│   │   ├── result.html          # Prediction result
│   │   └── dashboard.html       # User health dashboard
│   ├── models.py                # Prediction model (DB)
│   ├── views.py                 # Backend logic
│   └── urls.py                  # URL routing
├── data_cleaning.py             # Dataset preprocessing script
├── model_training.py            # LightGBM training script
├── diabetes.csv                 # Pima Indian Diabetes dataset
├── requirements.txt             # Python dependencies
└── manage.py                    # Django management
```

---

## How It Works

### 1. User Input
Users provide 8 clinical parameters via the prediction form:
- Gender, Pregnancies (females only), Glucose, Blood Pressure
- Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

### 2. Data Preprocessing
- Input values are scaled using the pre-fitted `StandardScaler`
- Male users automatically have Pregnancies set to 0
- Input validation enforces medically realistic ranges for each field

### 3. Model Prediction
- The LightGBM model is loaded **once at server startup** (not per request) for instant predictions
- The model outputs a probability score
- If probability ≥ 0.16 (optimal threshold) → **Diabetic**
- If probability < 0.16 → **Non-Diabetic**

### 4. Result Storage
- Prediction result is saved to SQLite with a UTC timestamp
- Displayed in the user's local browser timezone on the dashboard

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/RC-15-coder/CINS-490.git
cd CINS-490
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Development Server
```bash
python manage.py runserver
```
Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

---

## Optional: Regenerate Model Files

If you want to retrain the model from scratch:
```bash
# Step 1 — Clean and preprocess the dataset
python data_cleaning.py

# Step 2 — Train the model and save .pkl files
python model_training.py
```

This regenerates `best_lgb_model.pkl` and `scaler.pkl` with identical results (fixed `random_state=42`).

---

## Key Files

| File | Purpose |
|---|---|
| `views.py` | Handles prediction logic, preprocessing, and result storage |
| `model_training.py` | Trains LightGBM, finds optimal threshold via ROC curve, saves model |
| `data_cleaning.py` | Cleans raw dataset, removes outliers, creates train/test splits |
| `predictor/ml_model/` | Contains trained model and scaler pickle files |

---

## Disclaimer

This application is an educational project. It is **not a substitute for professional medical advice**. Always consult a qualified healthcare professional for any health decisions.
