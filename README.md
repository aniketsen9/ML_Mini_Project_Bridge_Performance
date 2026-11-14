# Predicting Bridge Performance under Earthquakes with Supervised Learning

## 1. Project Objective

This project investigates the application of supervised machine learning to predict bridge performance and structural damage following earthquake events. Using empirical structural and seismological records, the objective is to build a classification model that accurately identifies whether a bridge sustains earthquake damage based on its age, the earthquake magnitude, and its distance from the seismic epicenter.

In this updated version, a **Random Forest Classifier** is implemented as the primary machine learning model, utilizing the original empirical dataset.

---

## 2. Original Dataset and Features

The model uses the original empirical dataset [`Final Project Code/Data/ds1.csv`](Final%20Project%20Code/Data/ds1.csv) without alteration, synthetic augmentation, or external data.

### Dataset Overview
- **Total records:** 100 historical bridge earthquake observations
- **Missing values:** 0 (clean dataset)
- **Class distribution:**
  - Class `0` (Undamaged bridges): 90 instances (90.0%)
  - Class `1` (Damaged bridges): 10 instances (10.0%)
  *(The dataset represents a realistic imbalanced distribution common in structural hazard engineering)*

### Input Features ($X$)
Exactly the following three predictors are used:
- `x2`: **Bridge Age** (years at the time of the earthquake)
- `x4`: **Earthquake Magnitude** (Richter scale)
- `x5`: **Distance from Epicenter** (miles between bridge and earthquake epicenter)

```python
X = df[["x2", "x4", "x5"]]
```

### Target Label ($y$)
- `y1`: **Bridge Damage Target Label**
  - `0`: Undamaged / No structural damage
  - `1`: Damaged / Structural failure observed

```python
y = df["y1"]
```

> **Data Integrity Note:** The target label `y1` is strictly excluded from the input feature matrix $X$. No external data (such as Peak Ground Acceleration / PGA) or invented columns have been introduced, and the original row ordering and feature values in `ds1.csv` are preserved.

---

## 3. Random Forest Methodology

The classifier is built using scikit-learn's `RandomForestClassifier`. Random Forest constructs an ensemble of randomized decision trees trained on bootstrap samples of the training data and aggregates their predictions via majority voting.

### Configuration
```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1
)
```

- **`n_estimators=200`**: An ensemble of 200 trees stabilizes variance and reduces overfitting.
- **`random_state=42`**: Ensures deterministic reproducibility of data splits, tree bootstrapping, and feature selections.
- **`class_weight="balanced"`**: Automatically adjusts sample weights inversely proportional to class frequencies ($w_c = \frac{N}{2 \times N_c}$). This compensates for the 9:1 class imbalance without requiring synthetic resampling (e.g., SMOTE) or altering original records.
- **`min_samples_leaf=2`**: Regularizes tree depth and prevents individual leaf nodes from memorizing isolated outliers.
- **`n_jobs=-1`**: Utilizes all available CPU cores for parallelized tree generation.

*Note:* Random Forest is the model implemented in this version. While Random Forests are well-suited for non-linear decision boundaries and imbalanced data, this implementation does not make unverified claims of outperforming previous models without a standardized, cross-validated multi-model benchmark.

---

## 4. Training and Testing Procedure

1. **Train/Test Split:** An 80:20 train/test split is applied using `random_state=42`.
2. **Stratification:** Splitting is stratified on `y` (`stratify=y`) to maintain the identical 90:10 class ratio across partitions:
   - **Training Set (80 samples):** 72 Undamaged (`y1=0`), 8 Damaged (`y1=1`)
   - **Testing Set (20 samples):** 18 Undamaged (`y1=0`), 2 Damaged (`y1=1`)
3. **Disjoint Partitioning & Leakage Prevention:** The index sets of training and testing data are strictly disjoint. The model is fitted **only** on the training set, and evaluated **strictly** on the held-out test data.
4. **No Resampling Alteration:** No external rebalancing or resampling is applied to the raw data; class weighting within the loss function handles class skew naturally.

---

## 5. Evaluation Metrics

Evaluated strictly on the held-out test set of 20 records (18 Undamaged, 2 Damaged). Because the test set contains only two damaged examples, damaged-class metrics are highly sensitive to a single prediction and should be interpreted cautiously:

| Metric | Score | Note / Scope |
| :--- | :--- | :--- |
| **Overall Accuracy** | **95.00%** | 19 out of 20 test records correctly predicted |
| **Class 0 (Undamaged) Precision** | **94.74%** | 18 / 19 predicted undamaged were truly undamaged |
| **Class 0 (Undamaged) Recall** | **100.00%** | 18 / 18 actual undamaged bridges correctly identified |
| **Class 0 (Undamaged) F1-Score** | **0.9730** | Harmonic mean for the majority undamaged class |
| **Class 1 (Damaged) Precision** | **100.00%** | 1 / 1 predicted damaged bridge was truly damaged (0 false alarms) |
| **Class 1 (Damaged) Recall** | **50.00%** | 1 / 2 actual damaged bridges identified on test split |
| **Class 1 (Damaged) F1-Score** | **0.6667** | Harmonic mean for critical positive class |
| **Macro Average F1-Score** | **0.8198** | Unweighted average across both classes |
| **Weighted Average F1-Score** | **0.9423** | Support-weighted average across classes |

### Confusion Matrix
```
                    Predicted Undamaged (0)    Predicted Damaged (1)
Actual Undamaged (0)          18                          0
Actual Damaged   (1)           1                          1
```
- **True Negatives (TN):** 18 (Undamaged correctly predicted)
- **False Positives (FP):** 0 (No false alarms)
- **False Negatives (FN):** 1 (One damaged bridge classified as undamaged)
- **True Positives (TP):** 1 (Damaged bridge correctly detected)

### Feature Importance (Mean Decrease in Impurity / Gini)
- **`x4` (Earthquake Magnitude):** **60.79%** — Primary predictor of structural bridge damage.
- **`x2` (Bridge Age):** **23.83%** — Secondary predictor; older bridges exhibit higher vulnerability.
- **`x5` (Distance to Epicenter):** **15.39%** — Attenuation factor; proximity to seismic focus increases hazard.

---

## 6. How to Install Dependencies

The implementation requires Python 3.8 or later and standard scientific packages.

Install all requirements using pip:
```bash
python3 -m pip install -r requirements.txt
```

If you use Anaconda and `python3` cannot import packages installed in another environment, activate the intended environment first, then run the command above. Alternatively, install the dependencies into the active Conda environment with:
```bash
conda install pandas numpy scikit-learn matplotlib seaborn
```

Declared dependencies in `requirements.txt`:
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`

---

## 7. How to Run the Program

Run commands with `python3` on macOS. The implementation provides flexible execution options depending on your working directory.

### Option A: From Repository Root
```bash
# Using the convenience runner
python run_random_forest.py

# Or invoking the script directly
python "Final Project Code/Code/random_forest.py"
```

### Option B: From the `Final Project Code/Code/` Directory
```bash
cd "Final Project Code/Code"

# Run the dedicated Random Forest script
python random_forest.py

# Or run the adapted prediction script
python predict_x2_x4_x5.py
```

### Optional Command-Line Arguments
```bash
# Specify custom data path or output location
python "Final Project Code/Code/random_forest.py" --data-path "Final Project Code/Data/ds1.csv" --output-dir "Final Project Code/Output/random_forest_x2_x4_x5"
```

---

## 8. Generated Output Files

All output artifacts are saved in [`Final Project Code/Output/random_forest_x2_x4_x5/`](Final%20Project%20Code/Output/random_forest_x2_x4_x5/) and also copied to [`Final Project Code/Output/`](Final%20Project%20Code/Output/):

1. **`confusion_matrix.png`**:
   High-resolution (300 DPI) annotated heatmap showing actual vs. predicted labels with numeric counts.
2. **`feature_importance.png`**:
   High-resolution bar chart showing the relative Gini importance of `x2` (Age), `x4` (Magnitude), and `x5` (Distance).
3. **`prediction_results.csv`**:
   Detailed test-set predictions recording:
   - `original_index`: Index of sample from original `ds1.csv`
   - `x2`, `x4`, `x5`: Feature inputs
   - `actual_label`: Ground truth binary label (`0` or `1`)
   - `actual_meaning`: Ground truth status (`Undamaged` or `Damaged`)
   - `predicted_label`: Predicted binary label (`0` or `1`)
   - `predicted_meaning`: Predicted status (`Undamaged` or `Damaged`)
   - `prediction_correct`: Boolean verification (`True` / `False`)
4. **`evaluation_metrics.csv`**:
   Structured tabular export of all evaluation metrics (overall accuracy, macro/weighted averages, and per-class precision, recall, and F1-score).
