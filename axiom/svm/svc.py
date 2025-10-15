# axiom/svm/svc.py
import numpy as np

class SVC:
    """
    Support Vector Machine (SVM) classifier from first principles.
    Finds the optimal hyperplane that maximally separates classes.
    Uses hinge loss and L2 regularization.
    """
    
    def __init__(self, C=1.0, learning_rate=0.001, max_iters=1000, tol=1e-4):
        """
        Parameters:
        C : float - Regularization parameter
                  Smaller C = larger margin, more misclassifications allowed
                  Larger C = smaller margin, fewer misclassifications allowed
        learning_rate : float - Step size for gradient descent
        max_iters : int - Maximum training iterations
        tol : float - Convergence tolerance
        """
        self.C = C
        self.lr = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.w = None  # Weight vector
        self.b = None  # Bias term
        self.support_vectors_ = None
    
    def _hinge_loss(self, X, y):
        """
        Hinge loss for SVM.
        Formula: L = max(0, 1 - y_i(w·x_i + b))
        
        Why hinge loss?
        - Only penalizes misclassified points and points within margin
        - Creates sparse solution (only support vectors matter)
        - Leads to maximum margin classifier
        """
        distances = 1 - y * (X @ self.w + self.b)
        return np.maximum(0, distances)
    
    def fit(self, X, y):
        """
        Train SVM using gradient descent on hinge loss.
        Objective: min (1/2)||w||² + C * Σ hinge_loss_i
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Convert labels from [0,1] to [-1,1] for SVM
        y = np.where(y == 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        
        # Training loop
        for i in range(self.max_iters):
            # Calculate hinge loss for all samples
            distances = 1 - y * (X @ self.w + self.b)
            hinge_losses = np.maximum(0, distances)
            
            # Calculate gradients
            dw = np.zeros_like(self.w)
            db = 0.0
            
            for j in range(n_samples):
                if distances[j] > 0:  # Misclassified or within margin
                    dw += -y[j] * X[j]
                    db += -y[j]
            
            # Add regularization gradient and average
            dw = self.w + (self.C / n_samples) * dw
            db = (self.C / n_samples) * db
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # Calculate total loss for convergence check
            total_loss = 0.5 * np.sum(self.w ** 2) + self.C * np.mean(hinge_losses)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {total_loss:.4f}")
            
            # Check convergence
            if i > 0 and abs(total_loss - prev_loss) < self.tol:
                print(f"Converged after {i} iterations")
                break
            
            prev_loss = total_loss
        
        # Identify support vectors (points with non-zero hinge loss)
        self._identify_support_vectors(X, y)
        
        return self
    
    def _identify_support_vectors(self, X, y):
        """Find support vectors - points that define the margin"""
        distances = 1 - y * (X @ self.w + self.b)
        # Support vectors are points with hinge loss > 0 (on or within margin)
        support_mask = distances >= 0
        self.support_vectors_ = X[support_mask]
        self.support_vector_labels_ = y[support_mask]
    
    def decision_function(self, X):
        """
        Calculate signed distance to hyperplane.
        Positive = class 1, Negative = class -1
        """
        if self.w is None:
            raise ValueError("Model must be fitted first")
        
        X = np.array(X)
        return X @ self.w + self.b
    
    def predict(self, X):
        """
        Predict class labels based on sign of decision function.
        """
        distances = self.decision_function(X)
        # Convert back from [-1,1] to [0,1] labels
        return (distances >= 0).astype(int)
    
    def get_margin(self):
        """Calculate margin width = 2 / ||w||"""
        if self.w is None:
            raise ValueError("Model not fitted")
        return 2.0 / np.sqrt(np.sum(self.w ** 2))
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
