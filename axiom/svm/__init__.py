# axiom/svm/__init__.py
"""
Support Vector Machines Module

SVM for classification and regression using maximum margin optimization.
"""

from .svc import SVC
from .svr import SVR

__all__ = [
    'SVC',
    'SVR'
]
