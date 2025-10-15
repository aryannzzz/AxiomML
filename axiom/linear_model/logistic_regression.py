# axiom/linear_model/logistic_regression.py
import numpy as np

class LogisticRegression:
    """
    Logistic Regression for binary classification from first principles.
    Uses the sigmoid function to model probabilities and cross-entropy loss.
    Models: P(y=1|x) = σ(w·x + b) where σ is the sigmoid function
    """
    
    def __init__(self, learning_rate=0.01, max_iters=1000, fit_intercept=True, tol=1e-4):
        """
        Parameters:
        learning_rate : float - Step size for gradient descent
        max_iters : int - Maximum training iterations
        fit_intercept : bool - Whether to include bias term
        tol : float - Tolerance for convergence checking
        """
        self.lr = learning_rate
        self.max_iters = max_iters
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.theta = None  # Parameters [bias, weight1, weight2, ...]
        self.loss_history = []  # Track loss during training
    
    def _sigmoid(self, z):
        """
        Sigmoid (logistic) function: maps any real number to (0, 1)
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Why sigmoid?
        - Outputs can be interpreted as probabilities
        - Smooth gradient (helps with optimization)
        - Bounded between 0 and 1
        """
        # Clip z to prevent overflow in exponential
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        """Add column of 1s for intercept term"""
        if self.fit_intercept:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X
    
    def _cross_entropy_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss function.
        Measures how well predicted probabilities match true labels.
        
        Formula: L = -[y·log(p) + (1-y)·log(1-p)]
        
        Why cross-entropy?
        - Stronger gradient when wrong (compared to MSE)
        - Aligns well with probability interpretation
        - Convex function (guaranteed convergence)
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate loss for each sample
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)
    
    def fit(self, X, y):
        """
        Train logistic regression using gradient descent.
        Minimizes cross-entropy loss to find optimal parameters.
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Add intercept term if needed
        X_b = self._add_intercept(X)
        n_samples, n_features = X_b.shape
        
        # Initialize parameters randomly (small values)
        # Random initialization breaks symmetry and helps convergence
        self.theta = np.random.randn(n_features) * 0.01
        
        # Gradient descent loop
        for i in range(self.max_iters):
            # Forward pass: compute predictions and probabilities
            linear_output = X_b @ self.theta  # z = w·x + b
            probabilities = self._sigmoid(linear_output)  # p = σ(z)
            
            # Compute gradient of cross-entropy loss
            # Derivative: ∇L = (1/n) * Xᵀ · (p - y)
            errors = probabilities - y
            gradient = (X_b.T @ errors) / n_samples
            
            # Update parameters (move opposite to gradient)
            self.theta -= self.lr * gradient
            
            # Calculate and store loss for monitoring
            loss = self._cross_entropy_loss(y, probabilities)
            self.loss_history.append(loss)
            
            # Check for convergence (if loss change is small)
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                print(f"Converged after {i} iterations")
                break
        
        # Extract coefficients for sklearn-like interface
        if self.fit_intercept:
            self.intercept_ = self.theta[0]
            self.coef_ = self.theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta
            
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities P(y=1|x) for each sample.
        Returns array of probabilities between 0 and 1.
        """
        if self.theta is None:
            raise ValueError("Model must be fitted first")
        
        X_b = self._add_intercept(X)
        linear_output = X_b @ self.theta
        return self._sigmoid(linear_output)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1) using probability threshold.
        
        Why threshold at 0.5?
        - Natural cutoff for binary classification
        - Maximizes accuracy when classes are balanced
        - Can be adjusted for imbalanced datasets
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def decision_function(self, X):
        """Get raw linear output (before sigmoid)"""
        X_b = self._add_intercept(X)
        return X_b @ self.theta
