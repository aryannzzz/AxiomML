# axiom/metrics/classification_metrics.py
"""
Classification Metrics - Evaluating Classifier Performance

Metrics to evaluate binary and multi-class classification models.
All implemented from first principles using only NumPy.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Accuracy: Fraction of correct predictions
    
    Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Range: [0, 1], higher is better
    
    When to use:
    - Balanced datasets (roughly equal class sizes)
    - When all classes are equally important
    
    When NOT to use:
    - Imbalanced datasets (can be misleading)
    - When false positives and false negatives have different costs
    
    Example:
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> accuracy_score(y_true, y_pred)
    0.8
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Confusion Matrix: Table showing prediction vs actual classes
    
    Structure (binary):
                Predicted
                 0    1
    Actual  0 [ TN   FP ]
            1 [ FN   TP ]
    
    Where:
    - TN (True Negative): Correctly predicted negative
    - FP (False Positive): Incorrectly predicted positive (Type I error)
    - FN (False Negative): Incorrectly predicted negative (Type II error)
    - TP (True Positive): Correctly predicted positive
    
    Returns: (n_classes, n_classes) array
    
    Why it's useful:
    - Shows where the model is getting confused
    - Foundation for other metrics (precision, recall, F1)
    - Helps identify systematic errors
    
    Example:
    >>> y_true = [0, 1, 0, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1, 1]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 1],
           [1, 2]])
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm


def precision_score(y_true, y_pred, pos_label=1, average='binary'):
    """
    Precision: Of all positive predictions, how many were correct?
    
    Formula: Precision = TP / (TP + FP)
    
    Range: [0, 1], higher is better
    
    Intuition:
    - "When the model says positive, how often is it right?"
    - Measures false positive rate
    - Important when false positives are costly
    
    When to use:
    - Spam detection (false positives annoy users)
    - Medical screening (false positives cause unnecessary procedures)
    - Any case where "crying wolf" is problematic
    
    Parameters:
    average : str
        'binary': for binary classification (uses pos_label)
        'macro': average precision across all classes
        'weighted': weighted average by class support
    
    Example:
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]  # 2 TP, 0 FP
    >>> precision_score(y_true, y_pred)
    1.0  # All positive predictions were correct
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if average == 'binary':
        # Binary classification
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        
        if tp + fp == 0:
            return 0.0  # No positive predictions
        
        return tp / (tp + fp)
    
    elif average == 'macro':
        # Multi-class: average precision across classes
        classes = np.unique(y_true)
        precisions = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        
        return np.mean(precisions)
    
    elif average == 'weighted':
        # Weighted by class support
        classes = np.unique(y_true)
        precisions = []
        weights = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            support = np.sum(y_true == cls)
            
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
            
            weights.append(support)
        
        return np.average(precisions, weights=weights)
    
    else:
        raise ValueError(f"Unknown average type: {average}")


def recall_score(y_true, y_pred, pos_label=1, average='binary'):
    """
    Recall (Sensitivity, True Positive Rate): Of all actual positives, how many did we find?
    
    Formula: Recall = TP / (TP + FN)
    
    Range: [0, 1], higher is better
    
    Intuition:
    - "Of all the actual positives, how many did we catch?"
    - Measures false negative rate
    - Important when missing positives is costly
    
    When to use:
    - Disease detection (missing a disease is very bad)
    - Fraud detection (missing fraud is costly)
    - Search engines (want to find all relevant results)
    
    Trade-off with Precision:
    - High recall → predict positive more often → more false positives
    - High precision → predict positive less often → more false negatives
    - Can't maximize both simultaneously
    
    Parameters:
    average : str
        'binary': for binary classification (uses pos_label)
        'macro': average recall across all classes
        'weighted': weighted average by class support
    
    Example:
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]  # 2 TP, 1 FN
    >>> recall_score(y_true, y_pred)
    0.667  # Found 2 out of 3 actual positives
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if average == 'binary':
        # Binary classification
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        if tp + fn == 0:
            return 0.0  # No actual positives
        
        return tp / (tp + fn)
    
    elif average == 'macro':
        # Multi-class: average recall across classes
        classes = np.unique(y_true)
        recalls = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
        
        return np.mean(recalls)
    
    elif average == 'weighted':
        # Weighted by class support
        classes = np.unique(y_true)
        recalls = []
        weights = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            support = np.sum(y_true == cls)
            
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
            
            weights.append(support)
        
        return np.average(recalls, weights=weights)
    
    else:
        raise ValueError(f"Unknown average type: {average}")


def f1_score(y_true, y_pred, pos_label=1, average='binary'):
    """
    F1 Score: Harmonic mean of precision and recall
    
    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Range: [0, 1], higher is better
    
    Why harmonic mean?
    - Harmonic mean penalizes extreme values
    - Forces balance between precision and recall
    - If either is very low, F1 is low
    
    When to use:
    - Want to balance precision and recall
    - Imbalanced datasets
    - Need single metric that considers both false positives and false negatives
    
    Interpretation:
    - F1 = 1.0: Perfect precision and recall
    - F1 = 0.5: Moderate performance
    - F1 < 0.5: Poor performance
    
    Parameters:
    average : str
        'binary': for binary classification (uses pos_label)
        'macro': average F1 across all classes
        'weighted': weighted average by class support
    
    Example:
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> f1_score(y_true, y_pred)
    0.8  # Balanced precision and recall
    """
    precision = precision_score(y_true, y_pred, pos_label, average)
    recall = recall_score(y_true, y_pred, pos_label, average)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def classification_report(y_true, y_pred, labels=None, target_names=None):
    """
    Generate text report showing main classification metrics.
    
    Returns a dictionary with precision, recall, F1, and support for each class.
    
    Example:
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 1, 2]
    >>> report = classification_report(y_true, y_pred)
    >>> print(report)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if labels is None:
        labels = np.unique(y_true)
    
    if target_names is None:
        target_names = [f"class {label}" for label in labels]
    
    report = {}
    
    for i, (label, name) in enumerate(zip(labels, target_names)):
        # Binary metrics for this class vs rest
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, pos_label=1)
        recall = recall_score(y_true_binary, y_pred_binary, pos_label=1)
        f1 = f1_score(y_true_binary, y_pred_binary, pos_label=1)
        support = np.sum(y_true == label)
        
        report[name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    
    # Overall metrics
    report['accuracy'] = accuracy_score(y_true, y_pred)
    report['macro avg'] = {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1-score': f1_score(y_true, y_pred, average='macro'),
        'support': len(y_true)
    }
    report['weighted avg'] = {
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1-score': f1_score(y_true, y_pred, average='weighted'),
        'support': len(y_true)
    }
    
    return report


def log_loss(y_true, y_pred_proba, eps=1e-15):
    """
    Log Loss (Cross-Entropy Loss): Penalizes confident wrong predictions heavily
    
    Formula: -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
    
    Range: [0, ∞), lower is better
    
    Why use log loss?
    - Evaluates probability predictions, not just class labels
    - Heavily penalizes confident but wrong predictions
    - Smooth and differentiable (good for optimization)
    
    When to use:
    - When model outputs probabilities
    - Want to evaluate calibration of probabilities
    - Comparing probabilistic classifiers
    
    Parameters:
    y_true : array - True binary labels (0 or 1)
    y_pred_proba : array - Predicted probabilities [0, 1]
    eps : float - Small value to clip probabilities (avoid log(0))
    
    Example:
    >>> y_true = [1, 0, 1, 0]
    >>> y_pred_proba = [0.9, 0.1, 0.8, 0.3]  # Good predictions
    >>> log_loss(y_true, y_pred_proba)
    0.178  # Low loss
    
    >>> y_pred_proba = [0.1, 0.9, 0.2, 0.7]  # Bad predictions
    >>> log_loss(y_true, y_pred_proba)
    1.897  # High loss
    """
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    
    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    
    # Binary cross-entropy
    loss = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    
    return np.mean(loss)


def balanced_accuracy_score(y_true, y_pred):
    """
    Balanced Accuracy: Average recall across classes
    
    Formula: (Recall_class0 + Recall_class1 + ...) / n_classes
    
    Range: [0, 1], higher is better
    
    Why use balanced accuracy?
    - Better than accuracy for imbalanced datasets
    - Treats all classes equally (even if dataset doesn't)
    - Each class contributes equally to the score
    
    When to use:
    - Imbalanced datasets (e.g., 90% negative, 10% positive)
    - When all classes are equally important
    - Fraud detection, disease screening, etc.
    
    Example (imbalanced):
    >>> y_true = [0]*90 + [1]*10  # 90% class 0, 10% class 1
    >>> y_pred = [0]*100           # Always predict class 0
    >>> accuracy_score(y_true, y_pred)
    0.9  # Looks good but misleading
    >>> balanced_accuracy_score(y_true, y_pred)
    0.5  # More realistic (missed all class 1)
    """
    return recall_score(y_true, y_pred, average='macro')


def matthews_corrcoef(y_true, y_pred):
    """
    Matthews Correlation Coefficient (MCC): Balanced measure even for imbalanced classes
    
    Formula: MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    Range: [-1, 1]
        +1: Perfect prediction
         0: Random prediction
        -1: Perfect disagreement
    
    Why use MCC?
    - Works well with imbalanced datasets
    - Takes all confusion matrix values into account
    - Returns meaningful score even when classes are very imbalanced
    
    When to use:
    - Highly imbalanced datasets
    - Binary classification evaluation
    - Want single metric that considers TN, FP, FN, TP
    
    Advantages:
    - Only metric that considers all four confusion matrix values
    - Symmetric: treats both classes fairly
    - Returns high score only if prediction is good in all four confusion matrix categories
    
    Example:
    >>> y_true = [1, 1, 0, 0, 1, 0]
    >>> y_pred = [1, 1, 0, 1, 1, 0]
    >>> matthews_corrcoef(y_true, y_pred)
    0.577  # Moderate correlation
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Compute confusion matrix values
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate MCC
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator
