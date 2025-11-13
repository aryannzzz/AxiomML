# axiom/preprocessing/polynomial_features.py
import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    """
    Generate polynomial features from existing features.
    Transforms [x₁, x₂] into [1, x₁, x₂, x₁², x₁x₂, x₂², ...] for polynomial regression.
    This allows linear models to learn nonlinear relationships.
    """
    
    def __init__(self, degree=2, include_bias=True):
        """
        Parameters:
        degree : int - Highest degree of polynomial features
        include_bias : bool - Whether to include the bias (intercept) column of 1s
        """
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None
    
    def fit(self, X):
        """
        Learn the number of input features.
        In polynomial features, 'fitting' just means remembering the input shape.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_input_features_ = X.shape[1]
        return self
    
    def transform(self, X):
        """
        Transform features to polynomial features.
        """
        if self.n_input_features_ is None:
            raise ValueError("Must call fit before transform")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_input_features_:
            raise ValueError(f"Expected {self.n_input_features_} features, got {n_features}")
        
        # Calculate which combinations of features we need
        combinations = []
        
        # Start with bias term if requested
        if self.include_bias:
            combinations.append([0] * n_features)  # Represents the intercept term
        
        # Generate combinations for each degree from 1 to self.degree
        for degree in range(1, self.degree + 1):
            # combinations_with_replacement gives us all ways to choose 'degree' features
            # with replacement (so we can have x₁², x₁x₂, x₂², etc.)
            for comb in combinations_with_replacement(range(n_features), degree):
                combinations.append(list(comb))
        
        # Create the polynomial feature matrix
        X_poly = np.ones((n_samples, len(combinations)))
        
        for i, comb in enumerate(combinations):
            # Skip the bias term (already set to 1s)
            if sum(comb) == 0 and self.include_bias:
                continue
                
            # Multiply the features according to the combination
            # For [0, 0, 1] we compute x₀⁰ * x₁⁰ * x₂¹ = x₂
            # For [0, 1, 1] we compute x₀⁰ * x₁¹ * x₂¹ = x₁x₂
            for feature_idx in comb:
                X_poly[:, i] *= X[:, feature_idx]
        
        self.n_output_features_ = X_poly.shape[1]
        return X_poly
    
    def fit_transform(self, X):
        """Convenience method: fit and transform in one step"""
        return self.fit(X).transform(X)


