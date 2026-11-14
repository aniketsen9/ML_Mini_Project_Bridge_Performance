#!/usr/bin/env python3
"""
Bridge Earthquake Damage Prediction using Random Forest Classifier.

This script replaces the legacy Logistic Regression, QDA, and KNN implementations
with a Random Forest Classifier trained on the original ds1.csv dataset using
features [x2, x4, x5] and binary damage target y1.

For modular usage and custom arguments, see random_forest.py.
"""

import os
import sys

# Ensure current script directory is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from random_forest import (
    load_and_validate_data,
    prepare_features_and_target,
    split_dataset,
    train_random_forest,
    evaluate_model,
    plot_and_save_confusion_matrix,
    plot_and_save_feature_importance,
    save_prediction_results,
    save_evaluation_metrics,
    main,
)

if __name__ == "__main__":
    print("[INFO] Executing Random Forest Classifier for features [x2, x4, x5] on ds1.csv...")
    main()
