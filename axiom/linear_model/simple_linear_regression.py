# axiom/linear_model/simple_linear_regression.py
import numpy as np

class SimpleLinearRegression:
    """
    Simple Linear Regression from first principles.
    Models the relationship between a SINGLE feature and a target variable.
    Equation: y = b + w*x
    Where: b = y-intercept, w = slope coefficient
    """
    
    def __init__(self):
        # Initialize model parameters to None - they don't exist until we fit the model
        self.slope = None      # w in our equation (also called coefficient)
        self.intercept = None  # b in our equation (also called bias)
    
    def fit(self, X, y):
        """
        Train the model using the training data.
        Uses the closed-form solution (normal equations) for simple linear regression.
        
        Parameters:
        X : array-like, shape (n_samples,) - Single feature values
        y : array-like, shape (n_samples,) - Target values
        """
        # Convert to numpy arrays for numerical computations
        X = np.array(X).flatten()  # Ensure X is 1D array
        y = np.array(y).flatten()  # Ensure y is 1D array
        
        # Calculate basic statistics needed for the formulas
        n = len(X)                    # Number of data points
        x_mean = np.mean(X)           # Mean of feature values
        y_mean = np.mean(y)           # Mean of target values
        
        # Calculate the slope (w) using the formula:
        # w = Σ[(x_i - x_mean) * (y_i - y_mean)] / Σ[(x_i - x_mean)²]
        # This measures how much y changes for a unit change in x
        numerator = np.sum((X - x_mean) * (y - y_mean))    # Covariance between X and y
        denominator = np.sum((X - x_mean) ** 2)            # Variance of X
        
        # Avoid division by zero - if all X values are same, slope is undefined
        if denominator == 0:
            raise ValueError("All X values are identical - cannot compute slope")
            
        self.slope = numerator / denominator
        
        # Calculate the intercept (b) using the formula:
        # b = y_mean - w * x_mean
        # This is where the regression line crosses the y-axis when x=0
        self.intercept = y_mean - self.slope * x_mean
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        X : array-like, shape (n_samples,) - Feature values to predict on
        
        Returns:
        y_pred : array, shape (n_samples,) - Predicted target values
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.array(X).flatten()
        
        # Apply the linear equation: y_pred = b + w*x
        return self.intercept + self.slope * X
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).
        Measures how well the model explains the variance in the target.
        R² = 1 - (SS_residual / SS_total)
        Where: SS_residual = sum of squared errors of our model
               SS_total = total variance in the target
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
