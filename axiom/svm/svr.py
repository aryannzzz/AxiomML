# axiom/svm/svr.py
import numpy as np

class SVR:
    """
    Support Vector Regression from first principles.
    Finds a "tube" (margin) around the data where errors are not penalized.
    Only points outside this epsilon-tube contribute to the loss.
    """
    
    def __init__(self, C=1.0, epsilon=0.1, learning_rate=0.001, max_iters=1000):
        """
        Parameters:
        C : float - Regularization parameter. 
                   Smaller C = wider margin, more tolerance for errors
                   Larger C = narrower margin, less tolerance for errors
        epsilon : float - Width of the epsilon-tube. 
                        Errors within this tube are not penalized.
        learning_rate : float - Step size for gradient descent
        max_iters : int - Maximum number of iterations for training
        """
        self.C = C
        self.epsilon = epsilon
        self.lr = learning_rate
        self.max_iters = max_iters
        self.w = None  # Weight vector
        self.b = None  # Bias term
        self.support_vectors_ = None  # Data points that define the margin
        self.support_vector_labels_ = None  # Their corresponding targets
    
    def _epsilon_insensitive_loss(self, y_true, y_pred):
        """
        Epsilon-insensitive loss function.
        Only penalizes errors larger than epsilon.
        
        Formula: L = max(0, |y_true - y_pred| - epsilon)
        
        This means:
        - If error is within [-epsilon, epsilon] → loss = 0
        - If error is outside this range → loss = |error| - epsilon
        """
        errors = y_true - y_pred
        return np.maximum(0, np.abs(errors) - self.epsilon)
    
    def fit(self, X, y):
        """
        Train SVR using gradient descent.
        We minimize: (1/2)*||w||² + C * Σ max(0, |y_i - (w·x_i + b)| - epsilon)
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        n_samples, n_features = X.shape
        
        # Initialize parameters randomly
        # Small random values help break symmetry in gradient descent
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        
        # Training loop
        for iteration in range(self.max_iters):
            # Calculate predictions for all samples
            predictions = X @ self.w + self.b
            
            # Calculate errors
            errors = y - predictions
            
            # Initialize gradients
            dw = np.zeros_like(self.w)
            db = 0.0
            
            # Count support vectors (points outside epsilon tube)
            n_support_vectors = 0
            
            # Calculate gradients for each sample
            for i in range(n_samples):
                error = errors[i]
                
                # Check if this point is a support vector (outside epsilon tube)
                if np.abs(error) > self.epsilon:
                    n_support_vectors += 1
                    
                    # Determine the sign of the error for gradient direction
                    # If error is positive (y > prediction), we need to increase prediction
                    # If error is negative (y < prediction), we need to decrease prediction
                    sign = 1 if error > 0 else -1
                    
                    # Update gradients
                    # For points outside the tube, gradient = -sign * x_i for weights
                    # and -sign for bias
                    dw += -sign * X[i]
                    db += -sign
            
            # Add regularization gradient: ∂(1/2||w||²)/∂w = w
            dw = self.w + (self.C / n_samples) * dw
            db = (self.C / n_samples) * db
            
            # Update parameters using gradient descent
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # Print progress occasionally
            if iteration % 100 == 0:
                loss = self._calculate_total_loss(X, y)
                print(f"Iteration {iteration}: Loss = {loss:.4f}, "
                      f"Support vectors = {n_support_vectors}/{n_samples}")
        
        # Store support vectors for reference
        self._identify_support_vectors(X, y)
        
        return self
    
    def _calculate_total_loss(self, X, y):
        """Calculate total loss: regularization + epsilon-insensitive loss"""
        predictions = X @ self.w + self.b
        regularization_loss = 0.5 * np.sum(self.w ** 2)
        epsilon_loss = self.C * np.sum(self._epsilon_insensitive_loss(y, predictions))
        return regularization_loss + epsilon_loss
    
    def _identify_support_vectors(self, X, y):
        """Identify which data points are support vectors"""
        predictions = X @ self.w + self.b
        errors = np.abs(y - predictions)
        
        # Support vectors are points outside the epsilon tube
        support_mask = errors > self.epsilon
        self.support_vectors_ = X[support_mask]
        self.support_vector_labels_ = y[support_mask]
    
    def predict(self, X):
        """Make predictions using the learned hyperplane"""
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return X @ self.w + self.b
    
    def get_margin(self):
        """Calculate the margin width (inverse of ||w||)"""
        if self.w is None:
            raise ValueError("Model not fitted")
        return 1.0 / np.sqrt(np.sum(self.w ** 2))
