# axiom/__init__.py
"""
AxiomML - Building Machine Learning from First Principles

An educational machine learning library implementing algorithms from scratch.
All implementations use only NumPy and core Python for maximum transparency.

Modules:
--------
linear_model : Linear regression and logistic regression
tree : Decision trees for classification and regression
ensemble : Random forests and ensemble methods
svm : Support vector machines
neighbors : K-nearest neighbors
naive_bayes : Naive Bayes classifiers
neural_networks : Deep learning building blocks
preprocessing : Data preprocessing utilities
metrics : Evaluation metrics

Example Usage:
--------------
>>> from axiom.linear_model import LogisticRegression
>>> from axiom.tree import DecisionTreeClassifier
>>> from axiom.ensemble import RandomForestRegressor
>>> from axiom.neural_networks import Dense, ReLU, Adam
"""

# Import submodules for easier access
from . import linear_model
from . import tree
from . import ensemble
from . import svm
from . import neighbors
from . import naive_bayes
from . import neural_networks
from . import preprocessing
from . import metrics

# Commonly used classes for convenience
from .linear_model import (
    SimpleLinearRegression,
    MultipleLinearRegression,
    PolynomialRegression,
    LogisticRegression
)

from .tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from .ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from .svm import (
    SVC,
    SVR
)

from .neighbors import (
    KNeighborsClassifier
)

from .naive_bayes import (
    GaussianNB
)

__version__ = '0.1.0'

__all__ = [
    # Submodules
    'linear_model',
    'tree',
    'ensemble',
    'svm',
    'neighbors',
    'naive_bayes',
    'neural_networks',
    'preprocessing',
    'metrics',
    
    # Linear models
    'SimpleLinearRegression',
    'MultipleLinearRegression',
    'PolynomialRegression',
    'LogisticRegression',
    
    # Trees
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    
    # Ensemble
    'RandomForestClassifier',
    'RandomForestRegressor',
    
    # SVM
    'SVC',
    'SVR',
    
    # Neighbors
    'KNeighborsClassifier',
    
    # Naive Bayes
    'GaussianNB',
]
