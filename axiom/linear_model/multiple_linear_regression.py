# axiom/linear_model/multiple_linear_regression.py
import numpy as np

class MultipleLinearRegression:
    """
    Multiple Linear Regression from first principles.
    Handles multiple features: y = b + w₁x₁ + w₂x₂ + ... + wₙxₙ
    Uses matrix operations for efficient computation with multiple features.
    """
    
    def __init__(self, fit_intercept=True):
        """
        Parameters:
        fit_intercept : bool - Whether to calculate the intercept term
        """
        self.fit_intercept = fit_intercept
        self.coefficients = None  # Will store [b, w₁, w₂, ..., wₙ] if fit_intercept=True
                                  # or [w₁, w₂, ..., wₙ] if fit_intercept=False
        self.intercept_ = None    # For sklearn-like API
        self.coef_ = None         # For sklearn-like API
    
    @property
    def intercept(self):
        """Property for easier access to intercept (alias for intercept_)"""
        return self.intercept_
    
    def _add_intercept(self, X):
        """
        Add a column of 1s to the feature matrix for the intercept term.
        This trick allows us to treat the intercept as another coefficient in matrix operations.
        """
        # np.ones creates a column of 1s, np.c_ concatenates along second axis (columns)
        return np.c_[np.ones((X.shape[0], 1)), X]
    
    def fit(self, X, y):
        """
        Train the model using the normal equation (closed-form solution).
        Formula: θ = (XᵀX)⁻¹Xᵀy
        Where θ is the vector of coefficients [b, w₁, w₂, ...]
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        # If X has only one feature, reshape it to 2D (n_samples, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Add intercept term if requested
        if self.fit_intercept:
            X_design = self._add_intercept(X)
        else:
            X_design = X
        
        # Check if we have enough samples for the number of features
        n_samples, n_features = X_design.shape
        if n_samples <= n_features:
            raise ValueError(f"More features ({n_features}) than samples ({n_samples})")
        
        try:
            # Normal Equation: θ = (XᵀX)⁻¹Xᵀy
            # XᵀX gives the covariance matrix of features
            covariance_matrix = X_design.T @ X_design  # Matrix multiplication
            
            # Inverse of covariance matrix - this can fail if matrix is singular
            # (happens when features are perfectly correlated)
            inv_covariance = np.linalg.inv(covariance_matrix)
            
            # Multiply by Xᵀy to get the coefficients
            self.coefficients = inv_covariance @ (X_design.T @ y)
            
        except np.linalg.LinAlgError:
            # If matrix inversion fails (singular matrix), use pseudo-inverse instead
            # This is more numerically stable but computationally expensive
            self.coefficients = np.linalg.pinv(X_design) @ y
        
        # Set sklearn-like attributes for easier access
        if self.fit_intercept:
            self.intercept_ = self.coefficients[0]
            self.coef_ = self.coefficients[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.coefficients
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        Formula: y_pred = Xθ (matrix multiplication)
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Add intercept term if our model has one
        if self.fit_intercept:
            X_design = self._add_intercept(X)
        else:
            X_design = X
        
        # Matrix multiplication: X_design (n_samples × n_features) 
        # @ coefficients (n_features × 1) = predictions (n_samples × 1)
        return X_design @ self.coefficients
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).
        Measures how well the model explains the variance in the target.
        R² = 1 - (SS_residual / SS_total)
        """
        y_pred = self.predict(X)
        y = np.array(y).flatten()
        
        # Sum of squared residuals (errors our model makes)
        ss_residual = np.sum((y - y_pred) ** 2)
        # Total sum of squares (total variance in target)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        
        # R² score: 1 = perfect fit, 0 = no better than mean prediction
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared
    
    def get_params(self, feature_names=None):
        """
        Return the model parameters in a readable format.
        Useful for interpreting the model.
        """
        if self.coefficients is None:
            return "Model not fitted yet"
        
        params = {}
        if self.fit_intercept:
            params['intercept'] = self.coefficients[0]
            start_idx = 1
        else:
            params['intercept'] = 0.0
            start_idx = 0
        
        # Add coefficients for each feature
        for i, coef in enumerate(self.coefficients[start_idx:]):
            feature_name = feature_names[i] if feature_names else f'feature_{i}'
            params[feature_name] = coef
        
        return params
