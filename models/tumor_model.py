"""
نظام الكشف عن الأورام الخبيثة
Tumor Detection System - ML & DL Models
مصادر الداتا: Breast Cancer Wisconsin (حقيقية) + Synthetic + CSV خارجي
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve)
import os
import warnings
warnings.filterwarnings('ignore')

# ===== FEATURE DEFINITIONS =====
FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "symmetry_mean",
    "fractal_dimension_mean"
]

FEATURE_LABELS_AR = [
    "متوسط نصف القطر",
    "متوسط الملمس",
    "متوسط النعومة",
    "متوسط التراص",
    "متوسط التقعر",
    "متوسط التناسق",
    "متوسط البعد الكسوري"
]

FEATURE_RANGES = {
    "radius_mean":           (6.0,  30.0),
    "texture_mean":          (9.0,  40.0),
    "smoothness_mean":       (0.05, 0.17),
    "compactness_mean":      (0.02, 0.35),
    "concavity_mean":        (0.0,  0.45),
    "symmetry_mean":         (0.10, 0.30),
    "fractal_dimension_mean":(0.05, 0.10),
}

# ===== DATA SOURCES =====

def load_real_dataset():
    """Load real Breast Cancer Wisconsin dataset from sklearn (569 samples)."""
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    # sklearn encodes: 0=malignant, 1=benign → flip to 1=malignant
    df['diagnosis'] = 1 - bc.target
    cols7 = ['mean radius','mean texture','mean smoothness',
             'mean compactness','mean concavity','mean symmetry','mean fractal dimension']
    X = df[cols7].values
    y = df['diagnosis'].values
    return X, y, "Breast Cancer Wisconsin (Real - 569 samples)"


def load_csv_dataset(csv_path):
    """
    Load external CSV file.
    Expected columns: radius_mean, texture_mean, smoothness_mean,
                      compactness_mean, concavity_mean, symmetry_mean,
                      fractal_dimension_mean, diagnosis (0=benign,1=malignant)
    """
    df = pd.read_csv(csv_path)
    # Flexible column mapping
    col_map = {
        'mean radius':   'radius_mean',
        'mean texture':  'texture_mean',
        'mean smoothness': 'smoothness_mean',
        'mean compactness': 'compactness_mean',
        'mean concavity': 'concavity_mean',
        'mean symmetry':  'symmetry_mean',
        'mean fractal dimension': 'fractal_dimension_mean',
    }
    df.rename(columns=col_map, inplace=True)
    # Handle diagnosis column variants
    if 'diagnosis' not in df.columns:
        for alt in ['target','label','class','Diagnosis']:
            if alt in df.columns:
                df.rename(columns={alt:'diagnosis'}, inplace=True)
                break
    if 'diagnosis' in df.columns:
        # Handle M/B string labels
        if df['diagnosis'].dtype == object:
            df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0,'malignant':1,'benign':0,
                                                    'Malignant':1,'Benign':0,'1':1,'0':0})
    X = df[FEATURE_NAMES].values
    y = df['diagnosis'].values if 'diagnosis' in df.columns else None
    n = len(X)
    return X, y, f"External CSV ({n} samples)"


def generate_synthetic_dataset(n_samples=1000, random_state=42):
    """Generate synthetic breast cancer-like dataset."""
    rng = np.random.RandomState(random_state)
    n_benign = n_samples // 2
    benign = np.column_stack([
        rng.normal(12.0, 2.0, n_benign),
        rng.normal(17.0, 3.0, n_benign),
        rng.normal(0.090, 0.012, n_benign),
        rng.normal(0.080, 0.020, n_benign),
        rng.normal(0.045, 0.025, n_benign),
        rng.normal(0.170, 0.020, n_benign),
        rng.normal(0.062, 0.005, n_benign),
    ])
    n_malign = n_samples - n_benign
    malign = np.column_stack([
        rng.normal(17.5, 3.5, n_malign),
        rng.normal(21.0, 4.0, n_malign),
        rng.normal(0.103, 0.014, n_malign),
        rng.normal(0.145, 0.040, n_malign),
        rng.normal(0.160, 0.060, n_malign),
        rng.normal(0.205, 0.025, n_malign),
        rng.normal(0.064, 0.006, n_malign),
    ])
    X = np.vstack([benign, malign])
    y = np.array([0]*n_benign + [1]*n_malign)
    for i, feat in enumerate(FEATURE_NAMES):
        lo, hi = FEATURE_RANGES[feat]
        X[:, i] = np.clip(X[:, i], lo, hi)
    idx = rng.permutation(len(y))
    return X[idx], y[idx], f"Synthetic ({n_samples} samples)"


def load_combined_dataset(csv_path=None):
    """
    Load real + synthetic (+ optional CSV) and combine them.
    Returns X, y, source_description
    """
    # 1) Real data
    X_real, y_real, desc_real = load_real_dataset()

    # 2) Synthetic data
    X_syn, y_syn, desc_syn = generate_synthetic_dataset(800)

    # 3) Optional CSV
    if csv_path and os.path.exists(csv_path):
        try:
            X_csv, y_csv, desc_csv = load_csv_dataset(csv_path)
            if y_csv is not None:
                X_all = np.vstack([X_real, X_syn, X_csv])
                y_all = np.concatenate([y_real, y_syn, y_csv])
                return X_all, y_all, f"{desc_real} + {desc_syn} + {desc_csv}"
        except Exception as e:
            print(f"Warning: Could not load CSV: {e}")

    X_all = np.vstack([X_real, X_syn])
    y_all = np.concatenate([y_real, y_syn])
    return X_all, y_all, f"{desc_real} + {desc_syn}"


# ===== MODEL TRAINER =====
class TumorDetectionSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
            "SVM":                 SVC(probability=True, kernel='rbf', random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.data_source = None
        self.data_stats = {}
        self.is_trained = False

    def train(self, csv_path=None):
        X, y, self.data_source = load_combined_dataset(csv_path)

        # Data stats for UI
        self.data_stats = {
            "total":     int(len(y)),
            "malignant": int(y.sum()),
            "benign":    int((y == 0).sum()),
            "source":    self.data_source,
        }

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        best_auc = 0
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            acc  = accuracy_score(y_test, y_pred)
            auc  = roc_auc_score(y_test, y_prob)
            cv   = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc').mean()
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            cm   = confusion_matrix(y_test, y_pred)

            self.metrics[name] = {
                "accuracy": round(acc * 100, 2),
                "auc":      round(auc, 4),
                "cv_auc":   round(cv, 4),
                "fpr":      fpr.tolist(),
                "tpr":      tpr.tolist(),
                "cm":       cm.tolist(),
            }
            if auc > best_auc:
                best_auc = auc
                self.best_model = model
                self.best_model_name = name

        self.is_trained = True
        return self.metrics

    def retrain_with_csv(self, csv_path):
        """Retrain all models using real + synthetic + uploaded CSV."""
        self.__init__()
        return self.train(csv_path=csv_path)

    def predict(self, features: list):
        if not self.is_trained:
            self.train()
        arr = np.array(features, dtype=float).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)
        label = int(self.best_model.predict(arr_scaled)[0])
        prob  = float(self.best_model.predict_proba(arr_scaled)[0][1])

        votes = {}
        for name, mdl in self.models.items():
            p = float(mdl.predict_proba(arr_scaled)[0][1])
            votes[name] = round(p * 100, 1)

        rf = self.models["Random Forest"]
        importance = dict(zip(FEATURE_NAMES, rf.feature_importances_.tolist()))

        return {
            "label":       label,
            "probability": round(prob * 100, 2),
            "model_used":  self.best_model_name,
            "votes":       votes,
            "importance":  importance,
            "data_source": self.data_source,
            "data_stats":  self.data_stats,
        }

    def get_metrics(self):
        if not self.is_trained:
            self.train()
        return {"metrics": self.metrics, "data_stats": self.data_stats}


# Singleton
_system = None

def get_system():
    global _system
    if _system is None:
        _system = TumorDetectionSystem()
        _system.train()
    return _system

def reset_system():
    global _system
    _system = None
