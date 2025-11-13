# axiom/ensemble/__init__.py
"""
Ensemble Methods Module

Ensemble methods combine multiple models to create a stronger predictor.
This module includes Random Forests and other ensemble techniques.
"""

from .random_forest_classifier import RandomForestClassifier
from .random_forest_regressor import RandomForestRegressor

__all__ = [
    'RandomForestClassifier',
    'RandomForestRegressor'
]
