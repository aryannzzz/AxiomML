# axiom/neural_networks/losses.py
import numpy as np

class Loss:
    """
    Base class for loss functions.
    Loss functions measure how well the model's predictions match the true values.
    They guide the learning process by quantifying prediction error.
    """
    
    def forward(self, y_true, y_pred):
        """Compute loss value (forward pass)"""
        raise NotImplementedError
    
    def backward(self, y_true, y_pred):
        """Compute gradient of loss w.r.t predictions (backward pass)"""
        raise NotImplementedError
    
    def __call__(self, y_true, y_pred):
        """Allow calling loss like a function"""
        return self.forward(y_true, y_pred)


class MSELoss(Loss):
    """
    Mean Squared Error Loss: L = (1/n) * Σ(y_true - y_pred)²
    
    Mathematical Properties:
    - Measures average squared difference between predictions and targets
    - Always non-negative
    - Penalizes large errors more than small ones (quadratic penalty)
    
    When to use:
    - Regression problems
    - When you want to penalize outliers heavily
    - When errors are roughly Gaussian distributed
    
    Advantages:
    - Smooth and differentiable everywhere
    - Convex for linear models (single global minimum)
    - Easy to optimize
    
    Drawbacks:
    - Very sensitive to outliers (squared term amplifies large errors)
    - Can lead to exploding gradients with bad initialization
    - Not robust to outliers in data
    
    Mathematical intuition:
    - Minimizing MSE is equivalent to maximum likelihood estimation
      under Gaussian noise assumption
    - Gradient increases linearly with error magnitude
    - Derivative: dL/dy_pred = -2/n * (y_true - y_pred)
    """
    
    def forward(self, y_true, y_pred):
        """
        Compute mean squared error.
        
        Formula: (1/n) * Σ(y_true - y_pred)²
        
        Parameters:
        y_true : array - Ground truth values
        y_pred : array - Predicted values
        
        Returns:
        float - Mean squared error
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of MSE w.r.t predictions.
        
        Derivative: dL/dy_pred = -2/n * (y_true - y_pred)
        
        Intuition:
        - Gradient points in direction to reduce error
        - Magnitude proportional to error size
        - Negative sign because we want to increase y_pred when it's too small
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        n = y_true.shape[0] if y_true.ndim > 0 else 1
        return -2 * (y_true - y_pred) / n


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss: L = -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
    
    Mathematical Properties:
    - Measures difference between two probability distributions
    - Derived from maximum likelihood estimation
    - Asymmetric penalty (different for false positives vs false negatives)
    
    When to use:
    - Binary classification (2 classes)
    - When outputs are probabilities (after sigmoid)
    - Logistic regression
    
    Advantages:
    - Stronger gradients than MSE for classification
    - Theoretically motivated (information theory)
    - Works well with sigmoid activation
    
    Drawbacks:
    - Requires predictions in (0, 1) range
    - Numerically unstable if predictions near 0 or 1
    - Not suitable for regression
    
    Why cross-entropy?:
    - Measures "surprise" - how unexpected the prediction is
    - Low loss when confident and correct
    - High loss when confident but wrong
    - Connects to information theory and KL divergence
    """
    
    def __init__(self, epsilon=1e-15):
        """
        Parameters:
        epsilon : float - Small constant for numerical stability
        """
        self.epsilon = epsilon
    
    def forward(self, y_true, y_pred):
        """
        Compute binary cross-entropy.
        
        Formula: -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        
        Numerical stability:
        - Clip predictions to [epsilon, 1-epsilon] to avoid log(0)
        
        Parameters:
        y_true : array - True labels (0 or 1)
        y_pred : array - Predicted probabilities [0, 1]
        
        Returns:
        float - Binary cross-entropy loss
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute binary cross-entropy
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        
        return loss
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of binary cross-entropy w.r.t predictions.
        
        Derivative: dL/dŷ = 1/n * (ŷ - y) / [ŷ(1-ŷ)]
        
        When combined with sigmoid:
        - The derivative simplifies to: (ŷ - y)
        - This is why sigmoid + BCE is the standard pairing
        
        Intuition:
        - Gradient larger when prediction is confident but wrong
        - Gradient smaller when prediction is correct
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        n = y_true.shape[0] if y_true.ndim > 0 else 1
        
        # Gradient of BCE w.r.t predictions
        gradient = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n
        
        return gradient


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy Loss: L = -1/n * ΣΣ y_ij * log(ŷ_ij)
    
    Mathematical Properties:
    - Generalization of binary cross-entropy to multiple classes
    - Measures difference between true and predicted distributions
    - Sum over all classes for each sample
    
    When to use:
    - Multi-class classification (>2 classes)
    - When classes are mutually exclusive (one-hot encoded)
    - Always with softmax activation in output layer
    
    Advantages:
    - Proper probabilistic interpretation
    - Strong gradients for confident wrong predictions
    - Theoretically sound (maximum likelihood)
    
    Why with softmax:
    - Softmax outputs sum to 1 (probability distribution)
    - Combined gradient is simply: (ŷ - y)
    - This elegant simplification makes training efficient
    
    Intuition:
    - Only the predicted probability of the true class matters
    - High loss if model assigns low probability to true class
    - Low loss if model is confident and correct
    """
    
    def __init__(self, epsilon=1e-15):
        """
        Parameters:
        epsilon : float - Small constant for numerical stability
        """
        self.epsilon = epsilon
    
    def forward(self, y_true, y_pred):
        """
        Compute categorical cross-entropy.
        
        Formula: -1/n * ΣΣ y_ij * log(ŷ_ij)
        
        Parameters:
        y_true : array - True labels, one-hot encoded (n_samples, n_classes)
        y_pred : array - Predicted probabilities (n_samples, n_classes)
        
        Returns:
        float - Categorical cross-entropy loss
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute categorical cross-entropy
        # Only sum over true classes (one-hot: most terms are 0)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
        
        return loss
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of categorical cross-entropy w.r.t predictions.
        
        Derivative: dL/dŷ = -y / ŷ
        
        When combined with softmax:
        - The combined derivative is: (ŷ - y)
        - Remarkably simple and efficient
        
        Intuition:
        - Gradient pushes probabilities toward true labels
        - Larger gradient when model is confident but wrong
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        n = y_true.shape[0] if y_true.ndim > 0 else 1
        
        # Gradient of CCE w.r.t predictions
        gradient = -y_true / y_pred / n
        
        return gradient


class HingeLoss(Loss):
    """
    Hinge Loss: L = 1/n * Σ max(0, 1 - y_true * y_pred)
    
    Mathematical Properties:
    - Used for "maximum-margin" classification
    - Penalizes predictions on wrong side of margin
    - Zero loss for correct predictions beyond margin
    
    When to use:
    - Support Vector Machines (SVMs)
    - When you want maximum margin classification
    - Binary classification with labels {-1, +1}
    
    Advantages:
    - Focuses on hard examples (near decision boundary)
    - Encourages confident correct predictions
    - Sparse gradients (many samples have zero gradient)
    
    Drawbacks:
    - Not differentiable at y_true * y_pred = 1
    - Requires labels to be {-1, +1}, not {0, 1}
    - Less probabilistic interpretation than cross-entropy
    
    Mathematical intuition:
    - Loss = 0 when y_true * y_pred >= 1 (correct with margin)
    - Loss increases linearly for violations
    - The "margin" of 1 ensures confident predictions
    - Derivative: -y_true if violated, else 0
    """
    
    def forward(self, y_true, y_pred):
        """
        Compute hinge loss.
        
        Formula: 1/n * Σ max(0, 1 - y_true * y_pred)
        
        Parameters:
        y_true : array - True labels {-1, +1}
        y_pred : array - Predicted values (typically from linear layer)
        
        Returns:
        float - Hinge loss
        
        Note: y_true should be -1 or +1, not 0 or 1
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute margin violations
        margin = 1 - y_true * y_pred
        
        # Hinge loss: max(0, margin)
        loss = np.mean(np.maximum(0, margin))
        
        return loss
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of hinge loss w.r.t predictions.
        
        Derivative:
        - -y_true if y_true * y_pred < 1 (margin violated)
        - 0 otherwise (correctly classified with margin)
        
        Intuition:
        - Only samples violating margin contribute to gradient
        - Gradient direction: push prediction toward correct side
        - Magnitude: constant (unlike cross-entropy)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        n = y_true.shape[0] if y_true.ndim > 0 else 1
        
        # Gradient: -y_true where margin violated, else 0
        margin = 1 - y_true * y_pred
        gradient = np.where(margin > 0, -y_true, 0) / n
        
        return gradient


class HuberLoss(Loss):
    """
    Huber Loss: Combines MSE and MAE for robustness.
    
    Formula:
    - L = 0.5 * (y_true - y_pred)² if |error| <= δ
    - L = δ * (|error| - 0.5δ) if |error| > δ
    
    Mathematical Properties:
    - Quadratic for small errors (like MSE)
    - Linear for large errors (like MAE)
    - Differentiable everywhere (unlike MAE)
    
    When to use:
    - Regression with outliers
    - When you want robustness to extreme values
    - When MSE is too sensitive to outliers
    
    Advantages:
    - Robust to outliers (linear penalty for large errors)
    - Smooth gradient (differentiable everywhere)
    - Combines best of MSE and MAE
    
    Why it works:
    - δ parameter controls transition point
    - Small δ: more robust (closer to MAE)
    - Large δ: less robust (closer to MSE)
    - Smooth transition prevents gradient issues
    
    Intuition:
    - Be gentle with outliers (linear penalty)
    - Be strict with small errors (quadratic penalty)
    - Best of both worlds for robust regression
    """
    
    def __init__(self, delta=1.0):
        """
        Parameters:
        delta : float - Threshold between quadratic and linear regions
        """
        self.delta = delta
    
    def forward(self, y_true, y_pred):
        """
        Compute Huber loss.
        
        Parameters:
        y_true : array - True values
        y_pred : array - Predicted values
        
        Returns:
        float - Huber loss
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        error = y_true - y_pred
        abs_error = np.abs(error)
        
        # Quadratic region: |error| <= delta
        quadratic = 0.5 * error ** 2
        
        # Linear region: |error| > delta
        linear = self.delta * (abs_error - 0.5 * self.delta)
        
        # Use quadratic for small errors, linear for large
        loss = np.where(abs_error <= self.delta, quadratic, linear)
        
        return np.mean(loss)
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of Huber loss w.r.t predictions.
        
        Derivative:
        - -(y_true - y_pred) if |error| <= δ (like MSE)
        - -δ * sign(error) if |error| > δ (like MAE)
        
        Intuition:
        - Small errors: gradient proportional to error (MSE-like)
        - Large errors: gradient is constant (MAE-like)
        - Smooth transition at δ
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        n = y_true.shape[0] if y_true.ndim > 0 else 1
        
        error = y_true - y_pred
        abs_error = np.abs(error)
        
        # Gradient in quadratic region
        grad_quadratic = -error
        
        # Gradient in linear region
        grad_linear = -self.delta * np.sign(error)
        
        # Combine based on error magnitude
        gradient = np.where(abs_error <= self.delta, grad_quadratic, grad_linear) / n
        
        return gradient


class MAELoss(Loss):
    """
    Mean Absolute Error Loss: L = 1/n * Σ|y_true - y_pred|
    
    Mathematical Properties:
    - Measures average absolute difference
    - Linear penalty for errors (vs quadratic for MSE)
    - Not differentiable at zero
    
    When to use:
    - Regression with outliers
    - When you want equal penalty for all errors
    - When median is better metric than mean
    
    Advantages:
    - Robust to outliers (linear vs quadratic penalty)
    - Easier to interpret (same units as target)
    - Less sensitive to extreme values
    
    Drawbacks:
    - Not differentiable at error = 0
    - Can have multiple minima (not strictly convex)
    - Slower convergence than MSE
    
    Intuition:
    - All errors equally bad (no extra penalty for large errors)
    - Optimizes for median, not mean
    - Better when outliers should not dominate loss
    """
    
    def forward(self, y_true, y_pred):
        """
        Compute mean absolute error.
        
        Formula: 1/n * Σ|y_true - y_pred|
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return np.mean(np.abs(y_true - y_pred))
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of MAE w.r.t predictions.
        
        Derivative: -sign(y_true - y_pred)
        
        Note: Technically undefined at error = 0, we use 0 by convention
        
        Intuition:
        - Constant gradient magnitude (unlike MSE)
        - Only direction matters, not error size
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        n = y_true.shape[0] if y_true.ndim > 0 else 1
        
        # Gradient: -sign of error
        error = y_true - y_pred
        gradient = -np.sign(error) / n
        
        return gradient


# Convenience dictionary for easy loss lookup
LOSSES = {
    'mse': MSELoss,
    'mean_squared_error': MSELoss,
    'mae': MAELoss,
    'mean_absolute_error': MAELoss,
    'binary_crossentropy': BinaryCrossEntropy,
    'bce': BinaryCrossEntropy,
    'categorical_crossentropy': CategoricalCrossEntropy,
    'cce': CategoricalCrossEntropy,
    'hinge': HingeLoss,
    'huber': HuberLoss
}


def get_loss(name):
    """
    Get loss function by name.
    
    Parameters:
    name : str or Loss - Name of loss or loss instance
    
    Returns:
    Loss instance
    
    Example:
    >>> loss = get_loss('mse')
    >>> loss = get_loss(MSELoss())
    """
    if isinstance(name, Loss):
        return name
    
    if isinstance(name, str):
        name = name.lower()
        if name in LOSSES:
            return LOSSES[name]()
        else:
            raise ValueError(f"Unknown loss: {name}. Choose from {list(LOSSES.keys())}")
    
    raise TypeError(f"Loss must be string or Loss instance, got {type(name)}")
