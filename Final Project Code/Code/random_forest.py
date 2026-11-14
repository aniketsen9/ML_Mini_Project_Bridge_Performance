#!/usr/bin/env python3
"""
Random Forest Classifier for Bridge Earthquake Damage Prediction.

Project: Predicting Bridge Performance under Earthquakes with Supervised Learning
Dataset: ds1.csv
Features:
    - x2: Bridge age (years)
    - x4: Earthquake magnitude
    - x5: Distance from earthquake epicenter (miles)
Target:
    - y1: Bridge damage indicator (0 = Undamaged, 1 = Damaged)

This script trains a Random Forest Classifier using stratified 80:20 train/test split,
evaluates performance across imbalanced classes, generates visualizations (confusion
matrix and feature importances), and saves test prediction results.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def resolve_paths(data_path_arg=None, output_dir_arg=None):
    """
    Resolve dataset and output paths robustly so the script can be executed
    from the Code directory, repository root, or any other working directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Resolve dataset path
    if data_path_arg:
        resolved_data_path = os.path.abspath(data_path_arg)
    else:
        candidate_paths = [
            os.path.join(script_dir, "..", "Data", "ds1.csv"),
            os.path.join(repo_dir, "Final Project Code", "Data", "ds1.csv"),
            os.path.join(os.getcwd(), "Final Project Code", "Data", "ds1.csv"),
            os.path.join(os.getcwd(), "Data", "ds1.csv"),
            os.path.join(os.getcwd(), "ds1.csv"),
        ]
        resolved_data_path = None
        for path in candidate_paths:
            if os.path.isfile(path):
                resolved_data_path = os.path.abspath(path)
                break
        if resolved_data_path is None:
            resolved_data_path = os.path.abspath(candidate_paths[0])

    # Resolve output directories
    if output_dir_arg:
        primary_output_dir = os.path.abspath(output_dir_arg)
    else:
        primary_output_dir = os.path.join(script_dir, "..", "Output", "random_forest_x2_x4_x5")

    secondary_output_dir = os.path.join(script_dir, "..", "Output")

    return resolved_data_path, primary_output_dir, secondary_output_dir


def load_and_validate_data(csv_path):
    """
    Load ds1.csv and validate columns, data types, and values safely.

    Expected columns:
        - x2: Bridge age
        - x4: Earthquake magnitude
        - x5: Distance from earthquake epicenter
        - y1: Bridge damage target label (0 = Undamaged, 1 = Damaged)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Error: Dataset not found at '{csv_path}'. "
            f"Please verify that 'ds1.csv' exists in 'Final Project Code/Data/'."
        )

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_columns = ["x2", "x4", "x5", "y1"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Error: Dataset is missing required columns: {missing_columns}. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Check for missing/null values
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        print("Warning: Missing values detected in dataset:")
        print(null_counts[null_counts > 0])
        # Handle safely without altering dataset meaning
        df = df.dropna(subset=required_columns)
        print(f"Remaining records after dropping missing rows: {len(df)}")

    # Verify target values
    unique_targets = sorted(df["y1"].unique())
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(
            f"Error: Target column 'y1' contains unexpected values: {unique_targets}. "
            f"Expected binary labels {{0, 1}}."
        )

    print(f"Dataset successfully loaded: {len(df)} records.")
    print("Column overview:")
    print("  - x2 (Bridge Age): min={:.1f}, mean={:.1f}, max={:.1f}".format(
        df["x2"].min(), df["x2"].mean(), df["x2"].max()
    ))
    print("  - x4 (Earthquake Magnitude): min={:.2f}, mean={:.2f}, max={:.2f}".format(
        df["x4"].min(), df["x4"].mean(), df["x4"].max()
    ))
    print("  - x5 (Distance to Epicenter): min={:.1f}, mean={:.1f}, max={:.1f}".format(
        df["x5"].min(), df["x5"].mean(), df["x5"].max()
    ))
    print("Target distribution (y1):")
    counts = df["y1"].value_counts().to_dict()
    print(f"  - Class 0 (Undamaged bridges): {counts.get(0, 0)} ({counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  - Class 1 (Damaged bridges):   {counts.get(1, 0)} ({counts.get(1, 0)/len(df)*100:.1f}%)")

    return df


def prepare_features_and_target(df):
    """
    Extract exact features X = ['x2', 'x4', 'x5'] and target y = 'y1'.
    Ensures target is strictly excluded from input features.
    """
    feature_cols = ["x2", "x4", "x5"]
    X = df[feature_cols].copy()
    y = df["y1"].copy()

    assert "y1" not in X.columns, "Data integrity error: target column 'y1' must NOT be in X"
    assert list(X.columns) == ["x2", "x4", "x5"], "Feature set must strictly be ['x2', 'x4', 'x5']"

    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified 80:20 train/test split to avoid data leakage and
    preserve minority class proportions in train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Verify disjoint splits (no record overlap)
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0, "Data leakage detected: train/test overlap!"

    print("\n" + "="*50)
    print("TRAIN / TEST SPLIT (Stratified 80:20, random_state=42)")
    print("="*50)
    print(f"Training set: {len(X_train)} samples")
    print(f"  - Undamaged (y1=0): {(y_train == 0).sum()}")
    print(f"  - Damaged   (y1=1): {(y_train == 1).sum()}")
    print(f"Testing set:  {len(X_test)} samples")
    print(f"  - Undamaged (y1=0): {(y_test == 0).sum()}")
    print(f"  - Damaged   (y1=1): {(y_test == 1).sum()}")

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier using project specification:
        - n_estimators=200
        - random_state=42
        - class_weight="balanced"
        - min_samples_leaf=2
        - n_jobs=-1
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    )

    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*50)
    print(f"Model Configuration: {rf}")
    rf.fit(X_train, y_train)
    print("Model training completed successfully.")

    return rf


def evaluate_model(model, X_test, y_test):
    """
    Evaluate Random Forest model on held-out test data.
    Computes accuracy, per-class metrics, macro F1, weighted F1,
    confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Per-class metrics
    # Class 0: Undamaged
    prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    # Class 1: Damaged
    prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Macro and weighted averages
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    weighted_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report_str = classification_report(
        y_test,
        y_pred,
        target_names=["Undamaged (0)", "Damaged (1)"],
        digits=4
    )

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "class_0_undamaged": {"precision": prec_0, "recall": rec_0, "f1": f1_0},
        "class_1_damaged": {"precision": prec_1, "recall": rec_1, "f1": f1_1},
        "macro_avg": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
        "weighted_avg": {"precision": weighted_prec, "recall": weighted_rec, "f1": weighted_f1},
        "classification_report": report_str,
    }

    print("\n" + "="*50)
    print("EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("="*50)
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nPer-Class Breakdown:")
    print("  Class 0 [Undamaged Bridges]:")
    print(f"    - Precision: {prec_0:.4f}")
    print(f"    - Recall:    {rec_0:.4f}")
    print(f"    - F1-Score:  {f1_0:.4f}")
    print("  Class 1 [Damaged Bridges]:")
    print(f"    - Precision: {prec_1:.4f}")
    print(f"    - Recall:    {rec_1:.4f}")
    print(f"    - F1-Score:  {f1_1:.4f}")
    print("\nAggregate Averages:")
    print(f"  Macro Precision:    {macro_prec:.4f}")
    print(f"  Macro Recall:       {macro_rec:.4f}")
    print(f"  Macro F1-Score:     {macro_f1:.4f}")
    print(f"  Weighted F1-Score:  {weighted_f1:.4f}")
    print("\nConfusion Matrix [rows: actual, cols: predicted]:")
    print(cm)
    print("\nClassification Report:\n" + report_str)

    return y_pred, metrics


def plot_and_save_confusion_matrix(cm, output_paths):
    """
    Plot and save the confusion matrix with numerical counts,
    clear axis labels, and color intensity.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Undamaged (0)", "Damaged (1)"],
        yticklabels=["Undamaged (0)", "Damaged (1)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix\nRandom Forest Classifier (ds1.csv)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Target Label", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Target Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        print(f"Saved confusion matrix plot to: {path}")
    plt.close()


def plot_and_save_feature_importance(model, feature_names, output_paths):
    """
    Extract feature importances for x2, x4, x5 and plot a clearly labeled bar chart.
    """
    importances = model.feature_importances_
    feature_labels = {
        "x2": "x2 (Bridge Age)",
        "x4": "x4 (Earthquake Magnitude)",
        "x5": "x5 (Distance to Epicenter)"
    }
    display_names = [feature_labels.get(f, f) for f in feature_names]

    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [display_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]

    plt.figure(figsize=(7, 5))
    colors = ["#2b5c8f", "#418ab3", "#6baed6"]
    bars = plt.bar(sorted_names, sorted_importances, color=colors, edgecolor="black", linewidth=1.2, width=0.55)

    # Annotate numerical percentages on top of bars
    for bar, imp in zip(bars, sorted_importances):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.015,
            f"{imp:.4f}\n({imp*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.title("Feature Importance\nRandom Forest Classifier (ds1.csv)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Input Features", fontsize=11, fontweight="bold")
    plt.ylabel("Gini Importance Score", fontsize=11, fontweight="bold")
    plt.ylim(0, max(sorted_importances) + 0.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        print(f"Saved feature importance plot to: {path}")
    plt.close()

    print("\nFeature Importances Summary:")
    for name, imp in zip(sorted_names, sorted_importances):
        print(f"  - {name}: {imp:.4f} ({imp*100:.2f}%)")


def save_prediction_results(X_test, y_test, y_pred, output_paths):
    """
    Save held-out test set prediction results to CSV.
    Columns:
        - original_index: dataset row index
        - x2: bridge age
        - x4: earthquake magnitude
        - x5: distance to epicenter
        - actual_label: true y1 (0 or 1)
        - actual_meaning: Undamaged or Damaged
        - predicted_label: predicted y1 (0 or 1)
        - predicted_meaning: Undamaged or Damaged
        - prediction_correct: boolean True/False
    """
    label_map = {0: "Undamaged", 1: "Damaged"}

    results_df = pd.DataFrame({
        "original_index": X_test.index,
        "x2": X_test["x2"].values,
        "x4": X_test["x4"].values,
        "x5": X_test["x5"].values,
        "actual_label": y_test.values,
        "actual_meaning": [label_map[y] for y in y_test.values],
        "predicted_label": y_pred,
        "predicted_meaning": [label_map[y] for y in y_pred],
        "prediction_correct": (y_test.values == y_pred)
    })

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results_df.to_csv(path, index=False)
        print(f"Saved prediction results to: {path}")

    return results_df


def save_evaluation_metrics(metrics, output_paths):
    """
    Save structured evaluation metrics to CSV.
    """
    summary_data = [
        {"metric": "Accuracy", "value": round(metrics["accuracy"], 4), "class": "Overall"},
        {"metric": "Macro Precision", "value": round(metrics["macro_avg"]["precision"], 4), "class": "Macro Average"},
        {"metric": "Macro Recall", "value": round(metrics["macro_avg"]["recall"], 4), "class": "Macro Average"},
        {"metric": "Macro F1-Score", "value": round(metrics["macro_avg"]["f1"], 4), "class": "Macro Average"},
        {"metric": "Weighted F1-Score", "value": round(metrics["weighted_avg"]["f1"], 4), "class": "Weighted Average"},
        {"metric": "Precision", "value": round(metrics["class_0_undamaged"]["precision"], 4), "class": "Class 0 (Undamaged)"},
        {"metric": "Recall", "value": round(metrics["class_0_undamaged"]["recall"], 4), "class": "Class 0 (Undamaged)"},
        {"metric": "F1-Score", "value": round(metrics["class_0_undamaged"]["f1"], 4), "class": "Class 0 (Undamaged)"},
        {"metric": "Precision", "value": round(metrics["class_1_damaged"]["precision"], 4), "class": "Class 1 (Damaged)"},
        {"metric": "Recall", "value": round(metrics["class_1_damaged"]["recall"], 4), "class": "Class 1 (Damaged)"},
        {"metric": "F1-Score", "value": round(metrics["class_1_damaged"]["f1"], 4), "class": "Class 1 (Damaged)"},
    ]
    metrics_df = pd.DataFrame(summary_data)

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        metrics_df.to_csv(path, index=False)
        print(f"Saved evaluation metrics to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest Classifier for Bridge Earthquake Damage Prediction (ds1.csv)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to ds1.csv dataset file (optional, auto-detected if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to primary output directory (optional)"
    )
    args = parser.parse_args()

    print("="*60)
    print(" BRIDGE EARTHQUAKE DAMAGE PREDICTION: RANDOM FOREST CLASSIFIER")
    print("="*60)

    # 1. Resolve file paths
    csv_path, primary_out, secondary_out = resolve_paths(args.data_path, args.output_dir)

    # 2. Load and validate original dataset ds1.csv
    df = load_and_validate_data(csv_path)

    # 3. Extract exact features X = ['x2', 'x4', 'x5'] and target y = 'y1'
    X, y = prepare_features_and_target(df)

    # 4. Perform stratified 80:20 train/test split
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)

    # 5. Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)

    # 6. Evaluate model on held-out test data
    y_pred, metrics = evaluate_model(rf_model, X_test, y_test)

    # 7. Generate and save visualizations
    cm_paths = [
        os.path.join(primary_out, "confusion_matrix.png"),
        os.path.join(secondary_out, "confusion_matrix.png"),
    ]
    plot_and_save_confusion_matrix(metrics["confusion_matrix"], cm_paths)

    fi_paths = [
        os.path.join(primary_out, "feature_importance.png"),
        os.path.join(secondary_out, "feature_importance.png"),
    ]
    plot_and_save_feature_importance(rf_model, ["x2", "x4", "x5"], fi_paths)

    # 8. Save prediction results and evaluation metrics
    pred_paths = [
        os.path.join(primary_out, "prediction_results.csv"),
        os.path.join(secondary_out, "prediction_results.csv"),
    ]
    save_prediction_results(X_test, y_test, y_pred, pred_paths)

    metrics_paths = [
        os.path.join(primary_out, "evaluation_metrics.csv"),
        os.path.join(secondary_out, "evaluation_metrics.csv"),
    ]
    save_evaluation_metrics(metrics, metrics_paths)

    print("\n" + "="*60)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Primary output files saved to: {primary_out}")
    print(f"Secondary output files saved to: {secondary_out}")
    print("="*60)


if __name__ == "__main__":
    main()
