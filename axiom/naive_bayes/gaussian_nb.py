# axiom/naive_bayes/gaussian_nb.py
import numpy as np
from scipy.stats import norm

class GaussianNB:
    """
    Gaussian Naive Bayes classifier from first principles.
    Based on Bayes' theorem with independence assumption between features.
    P(y|X) ∝ P(y) * Π P(x_i|y)
    Assumes each feature follows normal distribution given the class.
    """
    
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = {}      # P(y) for each class
        self.class_means_ = {}       # μ for each feature in each class
        self.class_vars_ = {}        # σ² for each feature in each class
        self.epsilon = 1e-9          # Small value to avoid division by zero
    
    def fit(self, X, y):
        """
        Train Naive Bayes by calculating statistics for each class.
        For each class, compute:
        - Prior probability P(class)
        - Mean of each feature
        - Variance of each feature
        """
        X = np.array(X)
        y = np.array(y).flatten()
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        for cls in self.classes_:
            # Get samples belonging to this class
            X_cls = X[y == cls]
            n_cls_samples = len(X_cls)
            
            # Calculate prior probability: P(class) = (# in class) / (total samples)
            self.class_priors_[cls] = n_cls_samples / n_samples
            
            # Calculate mean and variance for each feature in this class
            # These define the Gaussian distribution for P(feature|class)
            self.class_means_[cls] = np.mean(X_cls, axis=0)
            self.class_vars_[cls] = np.var(X_cls, axis=0)
            
            # Add epsilon to variances to avoid division by zero
            self.class_vars_[cls] += self.epsilon
        
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        """
        Gaussian Probability Density Function.
        Calculates P(x|mean, variance) using normal distribution.
        
        Formula: (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        
        Why Gaussian assumption?
        - Works well for continuous features
        - Only need to store means and variances
        - Computationally efficient
        """
        coefficient = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coefficient * exponent
    
    def _calculate_log_likelihood(self, x, cls):
        """
        Calculate log P(x|cls) = Σ log P(x_i|cls)
        Uses log probabilities to avoid numerical underflow.
        
        Why log probabilities?
        - Multiplying many small probabilities can cause underflow
        - Log turns multiplication into addition
        - Maintains the same ordering for comparison
        """
        log_likelihood = 0.0
        
        # For each feature, add log probability
        for i in range(len(x)):
            prob = self._gaussian_pdf(x[i], self.class_means_[cls][i], self.class_vars_[cls][i])
            log_likelihood += np.log(prob + self.epsilon)  # Add epsilon to avoid log(0)
        
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities using Bayes' theorem.
        P(cls|x) ∝ P(cls) * P(x|cls)
        """
        if self.classes_ is None:
            raise ValueError("Model must be fitted first")
        
        X = np.array(X)
        n_samples = X.shape[0]
        probabilities = []
        
        for i in range(n_samples):
            x = X[i]
            class_probs = {}
            total_prob = 0.0
            
            # Calculate unnormalized probability for each class
            for cls in self.classes_:
                # Posterior ∝ prior * likelihood
                # In log space: log(posterior) = log(prior) + log(likelihood)
                log_prior = np.log(self.class_priors_[cls])
                log_likelihood = self._calculate_log_likelihood(x, cls)
                log_posterior = log_prior + log_likelihood
                
                # Convert back from log space
                class_probs[cls] = np.exp(log_posterior)
                total_prob += class_probs[cls]
            
            # Normalize to get proper probabilities
            normalized_probs = [class_probs.get(cls, 0) / total_prob for cls in self.classes_]
            probabilities.append(normalized_probs)
        
        return np.array(probabilities)
    
    def predict(self, X):
        """Predict class with highest probability"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_feature_importance(self, feature_idx):
        """
        Get feature importance based on class separation.
        Larger difference between class means = more important feature.
        """
        if len(self.classes_) != 2:
            raise ValueError("Feature importance only defined for binary classification")
        
        cls1, cls2 = self.classes_
        mean_diff = abs(self.class_means_[cls1][feature_idx] - self.class_means_[cls2][feature_idx])
        return mean_diff
