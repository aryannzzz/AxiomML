# axiom/linear_model/__init__.py
"""
Linear Models Module

Linear models for regression and classification.
Includes simple and multiple linear regression, polynomial regression, and logistic regression.
"""

from .simple_linear_regression import SimpleLinearRegression
from .multiple_linear_regression import MultipleLinearRegression
from .polynomial_regression import PolynomialRegression
from .logistic_regression import LogisticRegression

__all__ = [
    'SimpleLinearRegression',
    'MultipleLinearRegression',
    'PolynomialRegression',
    'LogisticRegression'
]
