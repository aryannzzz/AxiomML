# axiom/neural_networks/activations.py
import numpy as np

class Activation:
    """
    Base class for activation functions.
    Activation functions introduce non-linearity into neural networks,
    allowing them to learn complex patterns beyond linear relationships.
    """
    
    def forward(self, z):
        """Compute activation output (forward pass)"""
        raise NotImplementedError
    
    def backward(self, z):
        """Compute derivative for backpropagation (backward pass)"""
        raise NotImplementedError
    
    def __call__(self, z):
        """Allow calling activation like a function"""
        return self.forward(z)


class Sigmoid(Activation):
    """
    Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
    
    Mathematical Properties:
    - Output range: (0, 1)
    - Smooth S-shaped curve
    - Derivative: σ'(z) = σ(z) * (1 - σ(z))
    
    When to use:
    - Binary classification output layers
    - When you need probabilities between 0 and 1
    - Gate mechanisms in LSTMs
    
    Drawbacks:
    - Vanishing gradients for extreme values (z >> 0 or z << 0)
    - Outputs not zero-centered (can slow convergence)
    - Computationally expensive (exponential operation)
    
    Why it works:
    - Maps any real number to (0,1) range, useful for probabilities
    - Differentiable everywhere, crucial for gradient descent
    """
    
    def forward(self, z):
        """
        Compute sigmoid activation.
        
        Numerical stability trick:
        - For z >= 0: sigmoid(z) = 1 / (1 + exp(-z))
        - For z < 0: sigmoid(z) = exp(z) / (1 + exp(z))
        This prevents overflow from exp(large_number)
        """
        z = np.array(z, dtype=np.float64)
        
        # Stable computation to avoid overflow
        positive_mask = z >= 0
        negative_mask = ~positive_mask
        
        result = np.zeros_like(z, dtype=np.float64)
        
        # For positive z: 1 / (1 + exp(-z))
        result[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))
        
        # For negative z: exp(z) / (1 + exp(z))
        exp_z = np.exp(z[negative_mask])
        result[negative_mask] = exp_z / (1 + exp_z)
        
        return result
    
    def backward(self, z):
        """
        Compute gradient of sigmoid.
        
        Derivative: dσ/dz = σ(z) * (1 - σ(z))
        
        Intuition:
        - At z=0: gradient is maximum (0.25)
        - At extremes: gradient approaches 0 (vanishing gradient problem)
        """
        sigmoid_z = self.forward(z)
        return sigmoid_z * (1 - sigmoid_z)


class Tanh(Activation):
    """
    Hyperbolic tangent activation: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    
    Mathematical Properties:
    - Output range: (-1, 1)
    - Zero-centered (unlike sigmoid)
    - Derivative: tanh'(z) = 1 - tanh²(z)
    
    When to use:
    - Hidden layers (better than sigmoid due to zero-centering)
    - When you need outputs in range [-1, 1]
    - Recurrent neural networks
    
    Advantages over Sigmoid:
    - Zero-centered outputs help with gradient flow
    - Stronger gradients (derivative range: 0 to 1 vs sigmoid's 0 to 0.25)
    
    Drawbacks:
    - Still suffers from vanishing gradients (but less than sigmoid)
    - Computationally expensive
    
    Why it works:
    - Zero-centered outputs make optimization easier
    - Can be seen as a scaled sigmoid: tanh(z) = 2*sigmoid(2z) - 1
    """
    
    def forward(self, z):
        """
        Compute tanh activation.
        
        Numerical stability:
        Uses numpy's implementation which handles overflow internally
        """
        return np.tanh(z)
    
    def backward(self, z):
        """
        Compute gradient of tanh.
        
        Derivative: d(tanh)/dz = 1 - tanh²(z)
        
        Intuition:
        - Maximum gradient at z=0 (gradient = 1)
        - Approaches 0 at extremes (but slower than sigmoid)
        """
        tanh_z = self.forward(z)
        return 1 - tanh_z ** 2


class ReLU(Activation):
    """
    Rectified Linear Unit: ReLU(z) = max(0, z)
    
    Mathematical Properties:
    - Output range: [0, ∞)
    - Piecewise linear
    - Derivative: 1 if z > 0, else 0
    
    When to use:
    - Default choice for hidden layers in deep networks
    - Convolutional neural networks
    - Most modern architectures
    
    Advantages:
    - Computationally efficient (simple thresholding)
    - No vanishing gradient for positive values
    - Sparse activation (many neurons output 0)
    - Accelerates convergence vs sigmoid/tanh
    
    Drawbacks:
    - "Dying ReLU" problem: neurons can get stuck outputting 0
    - Not zero-centered
    - Unbounded output (can cause instability)
    
    Why it's the default:
    - Simple and effective
    - Biological inspiration (neurons either fire or don't)
    - Empirically works very well in deep networks
    """
    
    def forward(self, z):
        """
        Compute ReLU activation.
        
        Implementation: max(0, z)
        Vectorized using numpy's maximum function
        """
        return np.maximum(0, z)
    
    def backward(self, z):
        """
        Compute gradient of ReLU.
        
        Derivative:
        - 1 if z > 0
        - 0 if z <= 0
        
        Note: Technically undefined at z=0, we use 0 by convention
        """
        return (z > 0).astype(float)


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit: LeakyReLU(z) = max(αz, z)
    
    Mathematical Properties:
    - Output range: (-∞, ∞)
    - Piecewise linear with small negative slope
    - Derivative: 1 if z > 0, else α (typically α = 0.01)
    
    When to use:
    - When dying ReLU is a problem
    - As alternative to standard ReLU in deep networks
    - When you want small negative activations
    
    Advantages over ReLU:
    - Prevents dying ReLU (always has gradient)
    - Allows small negative values to propagate
    - Nearly as efficient as ReLU
    
    Why the leak matters:
    - Small negative slope (α) allows gradient flow for negative inputs
    - Neurons can "recover" if they get stuck in negative region
    - Empirically reduces dying ReLU problems
    
    Variants:
    - PReLU: α is learned during training
    - RReLU: α is randomized during training
    """
    
    def __init__(self, alpha=0.01):
        """
        Parameters:
        alpha : float - Slope for negative values (typically 0.01)
        """
        self.alpha = alpha
    
    def forward(self, z):
        """
        Compute Leaky ReLU activation.
        
        Implementation:
        - For z > 0: output = z
        - For z <= 0: output = α * z
        """
        return np.where(z > 0, z, self.alpha * z)
    
    def backward(self, z):
        """
        Compute gradient of Leaky ReLU.
        
        Derivative:
        - 1 if z > 0
        - α if z <= 0
        """
        return np.where(z > 0, 1, self.alpha)


class Softmax(Activation):
    """
    Softmax activation: σ(z)_i = e^(z_i) / Σ(e^(z_j))
    
    Mathematical Properties:
    - Converts vector of values to probability distribution
    - Output range: (0, 1) for each element
    - Sum of outputs = 1
    
    When to use:
    - Multi-class classification output layer (exclusive classes)
    - When you need probability distribution over classes
    - Always use in final layer for classification
    
    Why it works:
    - Exponential amplifies differences between values
    - Normalization ensures valid probability distribution
    - Differentiable, allowing gradient-based learning
    
    Mathematical intuition:
    - Larger inputs get exponentially larger probabilities
    - Small differences in input create large differences in output
    - "Soft" version of argmax (hence the name)
    
    Numerical stability:
    - Subtracting max(z) before exp prevents overflow
    - This doesn't change the result due to softmax's scale invariance
    - Critical for numerical stability with large values
    """
    
    def forward(self, z):
        """
        Compute softmax activation.
        
        Stability trick:
        - Subtract max(z) from all elements
        - softmax(z) = softmax(z - max(z)) (mathematically equivalent)
        - Prevents exp(large_number) overflow
        
        For batches:
        - Apply softmax row-wise (each sample independently)
        """
        z = np.array(z, dtype=np.float64)
        
        # Handle both 1D and 2D arrays (single sample vs batch)
        if z.ndim == 1:
            # Subtract max for numerical stability
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        else:
            # For batches: apply along last axis (features)
            # keepdims ensures broadcasting works correctly
            exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def backward(self, z):
        """
        Compute gradient of softmax.
        
        Derivative is complex:
        - For i=j: σ_i * (1 - σ_i)
        - For i≠j: -σ_i * σ_j
        - Results in Jacobian matrix
        
        In practice:
        - Usually combined with cross-entropy loss
        - The combined derivative is simply: σ(z) - y (predictions - targets)
        - This simplification is why softmax + cross-entropy is standard
        
        Note: This method returns the diagonal of the Jacobian for simplicity.
        Most implementations combine softmax with loss for efficiency.
        """
        softmax_z = self.forward(z)
        
        if softmax_z.ndim == 1:
            # Jacobian diagonal elements
            return softmax_z * (1 - softmax_z)
        else:
            # For batches, return diagonal elements
            return softmax_z * (1 - softmax_z)


class ELU(Activation):
    """
    Exponential Linear Unit: ELU(z) = z if z > 0 else α(e^z - 1)
    
    Mathematical Properties:
    - Output range: (-α, ∞)
    - Smooth everywhere (including at z=0)
    - Derivative: 1 if z > 0, else α*e^z
    
    When to use:
    - Alternative to ReLU with smoother behavior
    - When you want negative outputs with saturation
    - Deep networks where vanishing gradient is a concern
    
    Advantages:
    - No dying ReLU problem
    - Outputs close to zero mean (helps with gradient flow)
    - Smooth function (continuous first derivative)
    
    Drawbacks:
    - More computationally expensive than ReLU (exponential)
    - Requires tuning α parameter
    
    Why it helps:
    - Negative saturation pushes mean activations closer to zero
    - Reduces bias shift effect
    - Smooth gradient helps with optimization
    """
    
    def __init__(self, alpha=1.0):
        """
        Parameters:
        alpha : float - Controls negative saturation (typically 1.0)
        """
        self.alpha = alpha
    
    def forward(self, z):
        """
        Compute ELU activation.
        
        Implementation:
        - For z > 0: output = z
        - For z <= 0: output = α(e^z - 1)
        """
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))
    
    def backward(self, z):
        """
        Compute gradient of ELU.
        
        Derivative:
        - 1 if z > 0
        - α*e^z if z <= 0
        """
        return np.where(z > 0, 1, self.alpha * np.exp(z))


# Convenience dictionary for easy activation lookup
ACTIVATIONS = {
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'leakyrelu': LeakyReLU,
    'softmax': Softmax,
    'elu': ELU
}


def get_activation(name):
    """
    Get activation function by name.
    
    Parameters:
    name : str or Activation - Name of activation or activation instance
    
    Returns:
    Activation instance
    
    Example:
    >>> act = get_activation('relu')
    >>> act = get_activation(ReLU())
    """
    if isinstance(name, Activation):
        return name
    
    if isinstance(name, str):
        name = name.lower()
        if name in ACTIVATIONS:
            return ACTIVATIONS[name]()
        else:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATIONS.keys())}")
    
    raise TypeError(f"Activation must be string or Activation instance, got {type(name)}")
