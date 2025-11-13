# axiom/metrics/__init__.py
"""
Metrics Module - Evaluating Model Performance

Comprehensive collection of metrics for classification and regression.
All implemented from first principles using only NumPy.

Classification Metrics:
-----------------------
- accuracy_score: Fraction of correct predictions
- precision_score: Of predicted positives, how many are correct
- recall_score: Of actual positives, how many did we find
- f1_score: Harmonic mean of precision and recall
- confusion_matrix: Table of predictions vs actuals
- classification_report: Comprehensive classification metrics
- balanced_accuracy_score: Accuracy adjusted for class imbalance
- matthews_corrcoef: Correlation between predictions and actuals
- log_loss: Cross-entropy loss for probability predictions

Regression Metrics:
-------------------
- mean_squared_error (MSE): Average squared error
- root_mean_squared_error (RMSE): Square root of MSE
- mean_absolute_error (MAE): Average absolute error
- r2_score: Proportion of variance explained (R²)
- adjusted_r2_score: R² adjusted for number of features
- mean_absolute_percentage_error (MAPE): Average percentage error
- median_absolute_error: Median of absolute errors (robust)
- max_error: Maximum prediction error
- explained_variance_score: Variance explained by predictions
- mean_squared_log_error (MSLE): MSE in log space
- regression_report: Comprehensive regression metrics

Usage Examples:
---------------
Classification:
>>> from axiom.metrics import accuracy_score, precision_score, f1_score
>>> y_true = [0, 1, 1, 0, 1]
>>> y_pred = [0, 1, 0, 0, 1]
>>> print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
>>> print(f"Precision: {precision_score(y_true, y_pred)}")
>>> print(f"F1 Score: {f1_score(y_true, y_pred)}")

Regression:
>>> from axiom.metrics import mean_squared_error, r2_score, mean_absolute_error
>>> y_true = [3.0, 2.5, 4.0, 5.0]
>>> y_pred = [2.8, 2.7, 3.8, 5.2]
>>> print(f"MSE: {mean_squared_error(y_true, y_pred)}")
>>> print(f"R²: {r2_score(y_true, y_pred)}")
>>> print(f"MAE: {mean_absolute_error(y_true, y_pred)}")
"""

# Classification metrics
from .classification_metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    log_loss,
    balanced_accuracy_score,
    matthews_corrcoef
)

# Regression metrics
from .regression_metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    adjusted_r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
    explained_variance_score,
    mean_squared_log_error,
    regression_report
)

__all__ = [
    # Classification
    'accuracy_score',
    'confusion_matrix',
    'precision_score',
    'recall_score',
    'f1_score',
    'classification_report',
    'log_loss',
    'balanced_accuracy_score',
    'matthews_corrcoef',
    
    # Regression
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'adjusted_r2_score',
    'mean_absolute_percentage_error',
    'median_absolute_error',
    'max_error',
    'explained_variance_score',
    'mean_squared_log_error',
    'regression_report'
]