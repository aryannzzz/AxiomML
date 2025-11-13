# axiom/metrics/regression_metrics.py
"""
Regression Metrics - Evaluating Regression Model Performance

Metrics to evaluate regression models predicting continuous values.
All implemented from first principles using only NumPy.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE): Average squared difference between predictions and actual
    
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    
    Range: [0, ∞), lower is better
    
    Properties:
    - Always positive
    - Heavily penalizes large errors (quadratic)
    - In same units as y² (not interpretable directly)
    - Differentiable (good for optimization)
    
    When to use:
    - Most common regression metric
    - When large errors are particularly bad
    - As loss function during training
    
    When NOT to use:
    - When outliers are present (very sensitive)
    - When you need interpretable units
    
    Example:
    >>> y_true = [3.0, 2.5, 4.0, 5.0]
    >>> y_pred = [2.8, 2.7, 3.8, 5.2]
    >>> mean_squared_error(y_true, y_pred)
    0.0225  # Small MSE = good predictions
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE): Square root of MSE
    
    Formula: RMSE = √(MSE) = √[(1/n) * Σ(y_true - y_pred)²]
    
    Range: [0, ∞), lower is better
    
    Why use RMSE over MSE?
    - In same units as y (interpretable!)
    - "Average prediction error" in original units
    - Still penalizes large errors more than small ones
    
    When to use:
    - When you want interpretable error magnitude
    - Comparing models (same scale as target)
    - Reporting to non-technical audience
    
    Interpretation:
    - RMSE = 5: On average, predictions are off by 5 units
    - Lower RMSE = better model
    
    Example:
    >>> y_true = [100, 200, 300, 400]  # House prices in $1000s
    >>> y_pred = [110, 190, 310, 390]
    >>> root_mean_squared_error(y_true, y_pred)
    10.0  # Predictions are off by $10k on average
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE): Average absolute difference
    
    Formula: MAE = (1/n) * Σ|y_true - y_pred|
    
    Range: [0, ∞), lower is better
    
    Properties:
    - In same units as y (interpretable!)
    - Linear penalty (treats all errors equally)
    - More robust to outliers than MSE
    - Not differentiable at zero (harder to optimize)
    
    When to use:
    - When outliers are present
    - Want equal penalty for all errors
    - Need interpretable metric in original units
    
    MAE vs RMSE:
    - MAE: All errors weighted equally
    - RMSE: Large errors weighted more heavily
    - MAE ≤ RMSE always (equality when all errors same size)
    
    Example:
    >>> y_true = [10, 20, 30, 40]
    >>> y_pred = [12, 18, 32, 38]
    >>> mean_absolute_error(y_true, y_pred)
    2.0  # Average error is 2 units
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    R² Score (Coefficient of Determination): Proportion of variance explained
    
    Formula: R² = 1 - (SS_res / SS_tot)
    Where:
        SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
        SS_tot = Σ(y_true - ȳ)²       (total sum of squares)
    
    Range: (-∞, 1], higher is better
        1.0: Perfect predictions
        0.0: Model is as good as predicting the mean
        < 0: Model is worse than predicting the mean
    
    Interpretation:
    - R² = 0.8: Model explains 80% of variance in target
    - R² = 0.0: Model explains none of the variance
    - R² < 0: Model is worse than just predicting mean
    
    When to use:
    - Standard metric for regression
    - Comparing models on same dataset
    - Understanding model quality
    
    When NOT to use:
    - Comparing models on different datasets (not comparable)
    - Non-linear relationships (can be misleading)
    
    Properties:
    - Scale-free (unitless)
    - Can be negative (worse than baseline)
    - Always ≤ 1
    
    Example:
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.0, 2.9, 4.1, 5.0]  # Good predictions
    >>> r2_score(y_true, y_pred)
    0.98  # Explains 98% of variance
    
    >>> y_pred = [3, 3, 3, 3, 3]  # Just predict mean
    >>> r2_score(y_true, y_pred)
    0.0  # Explains 0% of variance
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        # All y_true values are the same
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Adjusted R²: R² adjusted for number of features
    
    Formula: Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    Where:
        n = number of samples
        p = number of features
    
    Range: (-∞, 1], higher is better
    
    Why adjusted R²?
    - R² always increases when adding features (even useless ones)
    - Adjusted R² penalizes adding unhelpful features
    - Better for comparing models with different numbers of features
    
    When to use:
    - Comparing models with different numbers of features
    - Feature selection
    - Preventing overfitting
    
    Properties:
    - Can decrease when adding features (unlike R²)
    - More conservative than R²
    - Adjusted R² ≤ R² always
    
    Parameters:
    n_features : int - Number of features used in the model
    
    Example:
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 2.0, 2.9, 4.1, 5.0]
    >>> adjusted_r2_score(y_true, y_pred, n_features=2)
    0.97  # Slightly lower than R² due to penalty
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    
    if n - n_features - 1 == 0:
        return r2
    
    adjusted = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    
    return adjusted


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    """
    Mean Absolute Percentage Error (MAPE): Average percentage error
    
    Formula: MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    
    Range: [0, ∞), lower is better
    
    Properties:
    - Scale-independent (can compare across different scales)
    - Expressed as percentage (easy to interpret)
    - Undefined when y_true = 0
    
    When to use:
    - Comparing models on different scales
    - Reporting to business stakeholders (percentage is intuitive)
    - When relative error is more important than absolute
    
    When NOT to use:
    - When y_true can be zero or close to zero
    - When y_true has both positive and negative values
    - Outliers (dividing by small values amplifies errors)
    
    Interpretation:
    - MAPE = 5%: Predictions are off by 5% on average
    - MAPE < 10%: Good model
    - MAPE > 25%: Poor model
    
    Parameters:
    epsilon : float - Small value to avoid division by zero
    
    Example:
    >>> y_true = [100, 200, 300]
    >>> y_pred = [110, 190, 310]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    6.67  # Predictions are off by ~6.67% on average
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Avoid division by zero
    y_true = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return mape


def median_absolute_error(y_true, y_pred):
    """
    Median Absolute Error: Median of absolute errors
    
    Formula: MedAE = median(|y_true - y_pred|)
    
    Range: [0, ∞), lower is better
    
    Why use median?
    - Very robust to outliers (more than MAE)
    - Not influenced by extreme values
    - Good when error distribution is skewed
    
    When to use:
    - When outliers are present
    - Want robust metric
    - Skewed error distributions
    
    MAE vs MedAE:
    - MAE: Sensitive to all errors equally
    - MedAE: Only cares about median (ignores extremes)
    - MedAE ≤ MAE typically
    
    Example:
    >>> y_true = [1, 2, 3, 4, 100]  # One outlier
    >>> y_pred = [1.1, 2.1, 2.9, 4.1, 50]  # Large error on outlier
    >>> mean_absolute_error(y_true, y_pred)
    10.04  # Inflated by outlier
    >>> median_absolute_error(y_true, y_pred)
    0.1  # Not affected by outlier
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    return np.median(np.abs(y_true - y_pred))


def max_error(y_true, y_pred):
    """
    Maximum Error: Worst prediction error
    
    Formula: Max Error = max(|y_true - y_pred|)
    
    Range: [0, ∞), lower is better
    
    Why use max error?
    - Shows worst-case performance
    - Important for safety-critical applications
    - Identifies if model has catastrophic failures
    
    When to use:
    - Safety-critical applications
    - Need to guarantee maximum error bounds
    - Identifying problematic predictions
    
    When NOT to use:
    - Dominated by single outlier
    - Not representative of typical performance
    
    Example:
    >>> y_true = [10, 20, 30, 40]
    >>> y_pred = [11, 19, 31, 50]  # Last one is way off
    >>> max_error(y_true, y_pred)
    10  # Worst prediction is off by 10
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    return np.max(np.abs(y_true - y_pred))


def explained_variance_score(y_true, y_pred):
    """
    Explained Variance Score: Proportion of variance explained
    
    Formula: EV = 1 - Var(y_true - y_pred) / Var(y_true)
    
    Range: (-∞, 1], higher is better
        1.0: Perfect predictions
        0.0: Predictions explain no variance
        < 0: Predictions worse than mean
    
    Similar to R²:
    - Both measure explained variance
    - EV uses variance, R² uses sum of squares
    - EV ignores systematic offsets (bias doesn't hurt score)
    
    When to use:
    - When systematic bias is acceptable
    - Comparing model predictions
    
    Example:
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.5, 2.5, 3.5, 4.5, 5.5]  # Systematic +0.5 offset
    >>> explained_variance_score(y_true, y_pred)
    1.0  # Still perfect (ignores bias)
    >>> r2_score(y_true, y_pred)
    0.9  # Lower (penalizes bias)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    var_diff = np.var(y_true - y_pred)
    var_true = np.var(y_true)
    
    if var_true == 0:
        return 1.0 if var_diff == 0 else 0.0
    
    return 1 - (var_diff / var_true)


def mean_squared_log_error(y_true, y_pred, epsilon=1e-10):
    """
    Mean Squared Logarithmic Error (MSLE): MSE in log space
    
    Formula: MSLE = (1/n) * Σ(log(y_true + 1) - log(y_pred + 1))²
    
    Range: [0, ∞), lower is better
    
    Why logarithmic?
    - Penalizes underestimation more than overestimation
    - Good for exponential growth (prices, populations)
    - Relative errors matter more than absolute
    
    When to use:
    - Targets with exponential trends
    - When relative differences matter
    - Don't want to penalize large values disproportionately
    
    When NOT to use:
    - When y_true or y_pred can be negative
    - When absolute errors are what matter
    
    Properties:
    - Only for positive targets
    - Asymmetric (underestimation penalized more)
    
    Parameters:
    epsilon : float - Small value to avoid log(0)
    
    Example:
    >>> y_true = [100, 200, 300]
    >>> y_pred = [110, 190, 310]
    >>> mean_squared_log_error(y_true, y_pred)
    0.0011  # Small error in log space
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("MSLE requires non-negative values")
    
    # Add 1 to avoid log(0), then compute log error
    log_true = np.log(y_true + 1 + epsilon)
    log_pred = np.log(y_pred + 1 + epsilon)
    
    return np.mean((log_true - log_pred) ** 2)


def regression_report(y_true, y_pred, n_features=None):
    """
    Generate comprehensive report of regression metrics.
    
    Returns dictionary with all major regression metrics.
    
    Example:
    >>> y_true = [3.0, 2.5, 4.0, 5.0]
    >>> y_pred = [2.8, 2.7, 3.8, 5.2]
    >>> report = regression_report(y_true, y_pred, n_features=2)
    >>> print(report)
    """
    report = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred),
        'Median AE': median_absolute_error(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred)
    }
    
    # Add MAPE if all values are positive
    if np.all(np.array(y_true) > 0):
        report['MAPE (%)'] = mean_absolute_percentage_error(y_true, y_pred)
    
    # Add Adjusted R² if n_features provided
    if n_features is not None:
        report['Adjusted R²'] = adjusted_r2_score(y_true, y_pred, n_features)
    
    return report
