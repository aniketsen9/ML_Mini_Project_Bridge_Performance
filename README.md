
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

*(The dataset represents a realistic imbalanced distribution common in structural hazard engineering.)*

### Input Features (X)

Exactly the following three predictors are used:

- `x2`: **Bridge Age** (years at the time of the earthquake)
- `x4`: **Earthquake Magnitude** (Richter scale)
- `x5`: **Distance from Epicenter** (miles between bridge and earthquake epicenter)

```python
X = df[["x2", "x4", "x5"]]
