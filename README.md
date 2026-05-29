# Car Owner Prediction System

A full-stack machine learning web application that predicts whether a used car has had 1st, 2nd, or 3rd owner based on vehicle features.

## Tech Stack
- **Backend:** Python, Flask, scikit-learn
- **ML Models:** Random Forest, KNN, Decision Tree, Logistic Regression, SVM
- **Data:** Pandas, NumPy
- **Frontend:** HTML, CSS, JavaScript

## Features
- Trains 5 classifiers and compares their accuracy
- REST API with /predict, /models, and /options endpoints
- Interactive frontend for real-time predictions
- Modular OOP architecture (DataHandler, FeatureEngineer, ModelTrainer)

## How to Run
```bash
pip install -r requirements.txt
python app.py
```
Then open http://localhost:5000
