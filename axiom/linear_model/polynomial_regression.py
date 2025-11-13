# axiom/linear_model/polynomial_regression.py
import numpy as np
from .multiple_linear_regression import MultipleLinearRegression
from ..preprocessing.polynomial_features import PolynomialFeatures

class PolynomialRegression:
    """
    Polynomial Regression by combining polynomial features with linear regression.
    This is NOT a separate algorithm - it's linear regression on transformed features!
    """
    
    def __init__(self, degree=2, fit_intercept=True):
        """
        Parameters:
        degree : int - Degree of the polynomial
        fit_intercept : bool - Whether to fit intercept term
        """
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.linear_model = MultipleLinearRegression(fit_intercept=fit_intercept)
    
    def fit(self, X, y):
        """
        Fit polynomial regression model.
        1. Transform features to polynomial features
        2. Fit linear regression on transformed features
        """
        # Step 1: Create polynomial features
        # This transforms [x] into [x, x², x³, ...] 
        X_poly = self.poly.fit_transform(X)
        
        # Step 2: Fit linear regression on polynomial features
        # The "magic" is that linear relationships in polynomial space 
        # correspond to nonlinear relationships in original space
        self.linear_model.fit(X_poly, y)
        
        return self
    
    def predict(self, X):
        """Make predictions using the polynomial regression model"""
        X_poly = self.poly.transform(X)
        return self.linear_model.predict(X_poly)
    
    def score(self, X, y):
        """R² score on polynomial features"""
        X_poly = self.poly.transform(X)
        return self.linear_model.score(X_poly, y)
