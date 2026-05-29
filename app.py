"""
Car Owner Prediction - Flask Backend
=====================================
Serves the ML models trained on car_data.csv and exposes a /predict endpoint.
"""

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = Flask(__name__, static_folder=".")

# ─────────────────────────────────────────────
# OOP Classes (mirrored from notebook)
# ─────────────────────────────────────────────

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded: {self.filepath} | Shape: {self.df.shape}")
        return self.df

    def clean(self):
        before = self.df.shape[0]
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        print(f"Cleaned: {before - self.df.shape[0]} rows removed | Remaining: {self.df.shape[0]}")
        return self.df


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}   # one encoder per column, kept for inference

    def encode(self, df, columns):
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        print(f"Encoded columns: {columns}")
        return df

    def encode_single(self, col, value):
        """Encode a single value using the fitted encoder for that column."""
        le = self.encoders.get(col)
        if le is None:
            raise ValueError(f"No encoder found for column '{col}'")
        val_str = str(value)
        classes = list(le.classes_)
        if val_str not in classes:
            # Return the closest or 0 as fallback
            return 0
        return int(le.transform([val_str])[0])

    def scale_fit(self, X_train, X_test):
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)
        return X_train_s, X_test_s

    def scale_transform(self, X):
        return self.scaler.transform(X)


class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(f"Trained: {type(self.model).__name__}")
        return self.model

    def predict(self, X):
        return self.model.predict(X)


# ─────────────────────────────────────────────
# Global pipeline state (loaded once at startup)
# ─────────────────────────────────────────────

class CarOwnerPipeline:
    """
    Trains multiple classifiers on car_data.csv and exposes predict().
    Target: Owner (0 = first owner, 1 = second owner, 3 = third owner)
    """

    CATEGORICAL_COLS = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
    FEATURE_COLS = [
        "Car_Name", "Year", "Selling_Price", "Present_Price",
        "Driven_kms", "Fuel_Type", "Selling_type", "Transmission"
    ]
    TARGET_COL = "Owner"

    MODELS = {
        "KNN":           KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":           SVC(probability=True, random_state=42),
    }

    def __init__(self, csv_path: str):
        self.fe = FeatureEngineer()
        self.trained_models: dict[str, ModelTrainer] = {}
        self.accuracies: dict[str, float] = {}
        self._train(csv_path)

    def _train(self, csv_path: str):
        handler = DataHandler(csv_path)
        df = handler.load()
        df = handler.clean()

        df = self.fe.encode(df, self.CATEGORICAL_COLS)

        X = df[self.FEATURE_COLS].values
        y = df[self.TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_s, X_test_s = self.fe.scale_fit(X_train, X_test)

        for name, model in self.MODELS.items():
            trainer = ModelTrainer(model)
            trainer.train(X_train_s, y_train)
            y_pred = trainer.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            self.trained_models[name] = trainer
            self.accuracies[name] = round(acc * 100, 2)
            print(f"  {name}: accuracy = {acc:.4f}")

        print("All models trained.")

    def predict(self, input_dict: dict, model_name: str) -> dict:
        """
        input_dict keys: Car_Name, Year, Selling_Price, Present_Price,
                         Driven_kms, Fuel_Type, Selling_type, Transmission
        Returns: { owner: int, probabilities: dict, model_accuracy: float }
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Unknown model '{model_name}'")

        row = [
            self.fe.encode_single("Car_Name",     input_dict["Car_Name"]),
            int(input_dict["Year"]),
            float(input_dict["Selling_Price"]),
            float(input_dict["Present_Price"]),
            int(input_dict["Driven_kms"]),
            self.fe.encode_single("Fuel_Type",    input_dict["Fuel_Type"]),
            self.fe.encode_single("Selling_type", input_dict["Selling_type"]),
            self.fe.encode_single("Transmission", input_dict["Transmission"]),
        ]
        X = np.array(row).reshape(1, -1)
        X_scaled = self.fe.scale_transform(X)

        trainer = self.trained_models[model_name]
        owner = int(trainer.predict(X_scaled)[0])

        # Probabilities (if model supports it)
        proba = {}
        if hasattr(trainer.model, "predict_proba"):
            probs = trainer.model.predict_proba(X_scaled)[0]
            classes = trainer.model.classes_
            proba = {int(c): round(float(p) * 100, 1) for c, p in zip(classes, probs)}

        return {
            "owner": owner,
            "probabilities": proba,
            "model_accuracy": self.accuracies[model_name],
            "all_accuracies": self.accuracies,
        }


# ─────────────────────────────────────────────
# Boot: train once
# ─────────────────────────────────────────────

CSV_PATH = os.path.join(os.path.dirname(__file__), "car_data.csv")
pipeline = CarOwnerPipeline(CSV_PATH)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        model_name = data.pop("model", "Random Forest")
        result = pipeline.predict(data, model_name)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/models", methods=["GET"])
def models():
    return jsonify({
        "models": list(pipeline.MODELS.keys()),
        "accuracies": pipeline.accuracies,
    })


@app.route("/options", methods=["GET"])
def options():
    """Return dropdown options for the frontend."""
    df = pd.read_csv(CSV_PATH)
    return jsonify({
        "Car_Name":    sorted(df["Car_Name"].unique().tolist()),
        "Fuel_Type":   df["Fuel_Type"].unique().tolist(),
        "Selling_type": df["Selling_type"].unique().tolist(),
        "Transmission": df["Transmission"].unique().tolist(),
        "Year":        {"min": int(df["Year"].min()), "max": int(df["Year"].max())},
        "Selling_Price": {"min": float(df["Selling_Price"].min()), "max": float(df["Selling_Price"].max())},
        "Present_Price": {"min": float(df["Present_Price"].min()), "max": float(df["Present_Price"].max())},
        "Driven_kms":  {"min": int(df["Driven_kms"].min()), "max": int(df["Driven_kms"].max())},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
