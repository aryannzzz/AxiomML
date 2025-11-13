# axiom/naive_bayes/__init__.py
"""
Naive Bayes Module

Probabilistic classifiers based on Bayes' theorem with naive independence assumption.
"""

from .gaussian_nb import GaussianNB

__all__ = [
    'GaussianNB'
]
