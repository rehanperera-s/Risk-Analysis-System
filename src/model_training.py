"""
Trains and evaluates churn prediction models.
Compares Logistic Regression, Random Forest, and Gradient Boosting.
Exports the best model and predictions with risk scores.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from imblearn.over_sampling import SMOTE

from src.feature_engineering import load_and_clean, prepare_features


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
}


def train_and_evaluate(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: {len(X_train)} -> {len(X_train_bal)} samples "
          f"(churn rate: {y_train.mean():.1%} -> {y_train_bal.mean():.1%})\n")
    results = {}

    for name, model in MODELS.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        results[name] = {"model": model, "metrics": metrics, "y_pred": y_pred, "y_prob": y_prob}

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return results


def select_best_model(results):
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    print(f"\nBest model: {best_name} (ROC-AUC: {results[best_name]['metrics']['roc_auc']:.4f})")
    return best_name, results[best_name]


def plot_model_comparison(results):
    metrics_df = pd.DataFrame({
        name: res["metrics"] for name, res in results.items()
    }).T

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("data/processed/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    return metrics_df


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(10, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc = res["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Churn Prediction Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/processed/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("data/processed/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    top_15 = feat_imp.tail(15)

    plt.figure(figsize=(10, 8))
    top_15.plot(kind="barh", color="#e74c3c", alpha=0.8)
    plt.title(f"Top 15 Churn Drivers — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("data/processed/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()


def save_model(model, scaler, encoders, feature_names, path="models"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, f"{path}/churn_model.pkl")
    joblib.dump(scaler, f"{path}/scaler.pkl")
    joblib.dump(encoders, f"{path}/encoders.pkl")
    joblib.dump(feature_names, f"{path}/feature_names.pkl")
    print(f"Model artifacts saved to {path}/")


def export_predictions(X_test, y_test, y_pred, y_prob, feature_names):
    """Export predictions with risk scores for Power BI."""
    customers_df = pd.read_csv("data/processed/customer_features.csv")
    test_indices = X_test.index

    output = customers_df.loc[test_indices].copy()
    output["churn_predicted"] = y_pred
    output["churn_probability"] = np.round(y_prob, 4)
    output["risk_segment"] = pd.cut(
        y_prob,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    output.to_csv("data/processed/churn_predictions.csv", index=False)
    print(f"Exported {len(output)} predictions to data/processed/churn_predictions.csv")

    full_model = joblib.load("models/churn_model.pkl")
    full_scaler = joblib.load("models/scaler.pkl")

    df_clean = load_and_clean()
    from src.feature_engineering import encode_categoricals
    df_encoded, _ = encode_categoricals(df_clean)
    X_all = df_encoded[[c for c in feature_names if c in df_encoded.columns]]
    X_all_scaled = pd.DataFrame(
        full_scaler.transform(X_all), columns=X_all.columns, index=X_all.index
    )

    all_probs = full_model.predict_proba(X_all_scaled)[:, 1]
    all_preds = full_model.predict(X_all_scaled)

    full_output = customers_df.copy()
    full_output["churn_predicted"] = all_preds
    full_output["churn_probability"] = np.round(all_probs, 4)
    full_output["risk_segment"] = pd.cut(
        all_probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    full_output.to_csv("data/processed/full_predictions_powerbi.csv", index=False)
    print(f"Exported full dataset ({len(full_output)} customers) for Power BI")

    return output


def run_training_pipeline():
    print("=== Loading and preparing features ===")
    df = load_and_clean()
    X_train, X_test, y_train, y_test, scaler, encoders, feature_names = prepare_features(df)

    print("\n=== Training models ===")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n=== Model comparison ===")
    plot_model_comparison(results)
    plot_roc_curves(results, y_test)

    best_name, best = select_best_model(results)
    plot_confusion_matrix(y_test, best["y_pred"], best_name)
    plot_feature_importance(best["model"], feature_names, best_name)

    print("\n=== Saving model ===")
    save_model(best["model"], scaler, encoders, feature_names)

    print("\n=== Exporting predictions ===")
    export_predictions(X_test, y_test, best["y_pred"], best["y_prob"], feature_names)

    print("\n=== Done ===")
    return results


if __name__ == "__main__":
    run_training_pipeline()
