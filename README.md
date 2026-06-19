# Surrogate Model Challenge

## Objective

The goal of this challenge was to improve the fidelity of an XGBoost surrogate model that approximates a Random Forest teacher model trained on the Diabetes dataset.

The evaluation metric was:

**Fidelity R² = R²(Teacher Predictions, Surrogate Predictions)**

The required threshold was **0.95**.

---

## Root Cause Analysis

### Issue 1: Surrogate Trained on Ground Truth Labels

The surrogate model was originally trained using the actual target values:

```python
target = sample["target"].values
```

However, the challenge evaluates how closely the surrogate reproduces the teacher model's predictions, not how accurately it predicts the ground truth labels.

This created a mismatch between:

* Training Objective → Predict true labels
* Evaluation Objective → Reproduce teacher predictions

### Fix

The surrogate target was changed to the teacher model's predictions:

```python
target = teacher.predict(sample[feature_names].values)
```

This aligns the training objective with the fidelity metric used for evaluation.

---

### Issue 2: Final Model Trained Only on Tuning Sample

GridSearchCV was performed on a representative sample of the training data for efficiency.

After selecting the best hyperparameters, the best estimator returned by GridSearchCV remained trained only on the sampled subset.

### Fix

After hyperparameter tuning, a new XGBoost model was created using the best parameters and retrained on the full training dataset using teacher predictions as targets.

This allowed the surrogate to learn from all available training examples while preserving the optimal hyperparameter configuration.

---

## Results

### Original Implementation

* Fidelity R²: **0.7827**
* Status: **FAIL**

### After Fix 1

* Fidelity R²: **0.9286**
* Status: **FAIL**

### Final Solution

* Fidelity R²: **0.9847**
* Fidelity RMSE: **6.3352**
* Status: **PASS**

---

## Key Insight

For surrogate modeling, the student model should be trained to imitate the teacher model rather than the original target labels. Aligning the training objective with the fidelity metric and retraining the best configuration on the full dataset significantly improved the surrogate's ability to reproduce the teacher's behavior.
