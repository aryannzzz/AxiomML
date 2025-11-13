# axiom/neural_networks/layers.py
import numpy as np

class Layer:
    """
    Base class for neural network layers.
    Layers transform inputs through learnable parameters and non-linear functions.
    Each layer must implement forward and backward passes for training.
    """
    
    def forward(self, inputs, training=True):
        """
        Forward pass: compute layer output.
        
        Parameters:
        inputs : array - Input data
        training : bool - Whether in training mode (affects dropout, batchnorm)
        
        Returns:
        array - Layer output
        """
        raise NotImplementedError
    
    def backward(self, grad_output):
        """
        Backward pass: compute gradients.
        
        Parameters:
        grad_output : array - Gradient of loss w.r.t layer output
        
        Returns:
        array - Gradient of loss w.r.t layer input
        """
        raise NotImplementedError
    
    def get_params(self):
        """Return layer parameters (weights, biases)"""
        return {}
    
    def get_grads(self):
        """Return gradients of parameters"""
        return {}
    
    def __call__(self, inputs, training=True):
        """Allow calling layer like a function"""
        return self.forward(inputs, training)


class Dense(Layer):
    """
    Fully Connected (Dense) Layer: y = xW + b
    
    Mathematical Properties:
    - Linear transformation of input
    - Each output connected to all inputs
    - Fundamental building block of neural networks
    
    Parameters:
    - W : (input_dim, output_dim) - Weight matrix
    - b : (output_dim,) - Bias vector
    
    When to use:
    - Most common layer type
    - After convolutional/recurrent layers
    - Output layers for classification/regression
    
    How it works:
    - Matrix multiplication followed by bias addition
    - Each neuron computes weighted sum of all inputs
    - Typically followed by activation function
    
    Why it's called "Dense":
    - Every input connected to every output (dense connectivity)
    - Contrast with sparse layers (e.g., convolutional)
    - Most parameters in typical networks are in dense layers
    
    Forward pass:
    - Output = Input @ Weights + Bias
    - Shape: (batch, input_dim) @ (input_dim, output_dim) = (batch, output_dim)
    
    Backward pass:
    - dL/dW = Input^T @ dL/dOutput
    - dL/db = sum(dL/dOutput, axis=0)
    - dL/dInput = dL/dOutput @ Weights^T
    """
    
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        Parameters:
        input_dim : int - Number of input features
        output_dim : int - Number of output features (neurons)
        use_bias : bool - Whether to include bias term
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # Will be initialized by initializer
        self.weights = None
        self.bias = None if use_bias else None
        
        # Cache for backward pass
        self.inputs = None
        
        # Gradients
        self.grad_weights = None
        self.grad_bias = None
    
    def initialize(self, initializer='xavier'):
        """
        Initialize weights and biases.
        
        Called separately to allow custom initialization strategies.
        See initializers.py for different initialization methods.
        """
        from .initializers import get_initializer
        init = get_initializer(initializer)
        
        self.weights = init((self.input_dim, self.output_dim))
        
        if self.use_bias:
            # Biases typically initialized to zero
            self.bias = np.zeros(self.output_dim)
    
    def forward(self, inputs, training=True):
        """
        Forward pass: compute y = xW + b
        
        Parameters:
        inputs : array of shape (batch_size, input_dim)
        
        Returns:
        outputs : array of shape (batch_size, output_dim)
        """
        # Cache inputs for backward pass
        self.inputs = inputs
        
        # Linear transformation: y = xW + b
        output = inputs @ self.weights
        
        if self.use_bias:
            output += self.bias
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass: compute gradients.
        
        Chain rule application:
        - dL/dW = X^T @ dL/dY
        - dL/db = sum(dL/dY)
        - dL/dX = dL/dY @ W^T
        
        Parameters:
        grad_output : array of shape (batch_size, output_dim)
                     Gradient of loss w.r.t layer output
        
        Returns:
        grad_input : array of shape (batch_size, input_dim)
                    Gradient of loss w.r.t layer input
        """
        # Gradient w.r.t weights: input^T @ grad_output
        # Shape: (input_dim, batch) @ (batch, output_dim) = (input_dim, output_dim)
        self.grad_weights = self.inputs.T @ grad_output
        
        # Gradient w.r.t bias: sum over batch dimension
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t input: grad_output @ weights^T
        # Shape: (batch, output_dim) @ (output_dim, input_dim) = (batch, input_dim)
        grad_input = grad_output @ self.weights.T
        
        return grad_input
    
    def get_params(self):
        """Return layer parameters"""
        params = {'weights': self.weights}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    
    def get_grads(self):
        """Return parameter gradients"""
        grads = {'weights': self.grad_weights}
        if self.use_bias:
            grads['bias'] = self.grad_bias
        return grads


class Dropout(Layer):
    """
    Dropout Layer: Randomly zero out activations during training.
    
    Mathematical Properties:
    - During training: randomly set activations to 0 with probability p
    - During inference: use all activations (no dropout)
    - Scale remaining activations by 1/(1-p) to maintain expected value
    
    When to use:
    - Prevent overfitting in deep networks
    - After dense or convolutional layers
    - Typically not used before output layer
    
    Why it works (regularization):
    - Forces network to learn redundant representations
    - Prevents co-adaptation of neurons
    - Each neuron must work independently
    - Ensemble effect: training multiple subnetworks
    
    Mathematical intuition:
    - Each forward pass trains a different subnetwork
    - Averaging over all possible subnetworks at test time
    - Approximate this by using all neurons with scaled activations
    
    Key insight:
    - Training: y = x * mask / (1-p), where mask ~ Bernoulli(1-p)
    - Testing: y = x (no dropout, already scaled during training)
    
    Why the scaling?:
    - Expected value during training: E[y] = x * (1-p) / (1-p) = x
    - Keeps expected activation magnitude constant
    - "Inverted dropout" - scale during training, not test
    
    Typical values:
    - p = 0.5 for hidden layers (dropout half the neurons)
    - p = 0.2-0.3 for input layers (less aggressive)
    - p = 0 for output layer (no dropout)
    """
    
    def __init__(self, dropout_rate=0.5):
        """
        Parameters:
        dropout_rate : float - Probability of dropping a neuron (0 to 1)
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, inputs, training=True):
        """
        Forward pass with dropout.
        
        Training mode:
        - Generate random binary mask
        - Zero out activations with probability p
        - Scale remaining by 1/(1-p)
        
        Inference mode:
        - Return inputs unchanged (no dropout)
        
        Parameters:
        inputs : array - Layer input
        training : bool - Whether in training mode
        
        Returns:
        array - Output with dropout applied (if training)
        """
        if not training or self.dropout_rate == 0:
            # No dropout during inference
            return inputs
        
        # Generate binary mask: 1 with prob (1-p), 0 with prob p
        # Each element independently sampled
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
        
        # Apply mask and scale to maintain expected value
        # Scaling by 1/(1-p) is "inverted dropout"
        output = inputs * self.mask / (1 - self.dropout_rate)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass: propagate gradients through dropout.
        
        Only neurons that were kept in forward pass get gradients.
        Same mask and scaling applied to gradients.
        
        Parameters:
        grad_output : array - Gradient w.r.t output
        
        Returns:
        array - Gradient w.r.t input
        """
        if self.mask is None:
            # No dropout was applied (inference mode)
            return grad_output
        
        # Apply same mask and scaling as forward pass
        grad_input = grad_output * self.mask / (1 - self.dropout_rate)
        
        return grad_input


class BatchNorm(Layer):
    """
    Batch Normalization: Normalize activations across batch dimension.
    
    Formula (training):
    - μ = mean(x, axis=0)              # Batch mean
    - σ² = var(x, axis=0)              # Batch variance
    - x_norm = (x - μ) / √(σ² + ε)    # Normalize
    - y = γ * x_norm + β               # Scale and shift
    
    Mathematical Properties:
    - Normalizes each feature independently across batch
    - Learns optimal scale (γ) and shift (β) parameters
    - Maintains running statistics for inference
    
    When to use:
    - After linear/convolutional layers, before activation
    - Deep networks (helps with vanishing gradients)
    - When you want faster training and better convergence
    
    Why it helps:
    - Reduces internal covariate shift
    - Allows higher learning rates
    - Reduces dependence on initialization
    - Acts as regularization (slight noise from batch statistics)
    - Smoother loss landscape
    
    Key benefits:
    1. Faster convergence (can use higher learning rates)
    2. Less sensitive to initialization
    3. Acts as implicit regularization
    4. Reduces gradient vanishing/explosion
    
    Training vs Inference:
    - Training: use batch statistics (mean, var from current batch)
    - Inference: use running statistics (moving average from training)
    - This difference is crucial for correct behavior
    
    Learnable parameters:
    - γ (gamma): scale parameter, initialized to 1
    - β (beta): shift parameter, initialized to 0
    
    Why scale and shift after normalizing?:
    - Normalization might remove useful information
    - γ and β let network learn optimal distribution
    - Can even undo normalization if needed (γ=σ, β=μ)
    """
    
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        """
        Parameters:
        num_features : int - Number of features to normalize
        momentum : float - Momentum for running mean/var (typically 0.9)
        epsilon : float - Small constant for numerical stability
        """
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(num_features)   # Scale
        self.beta = np.zeros(num_features)   # Shift
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward pass
        self.x_norm = None
        self.mean = None
        self.var = None
        self.x_centered = None
        self.std = None
        
        # Gradients
        self.grad_gamma = None
        self.grad_beta = None
    
    def forward(self, inputs, training=True):
        """
        Forward pass: normalize, scale, and shift.
        
        Training mode:
        - Compute batch statistics (mean, var)
        - Normalize using batch statistics
        - Update running statistics
        
        Inference mode:
        - Use running statistics (accumulated during training)
        - Deterministic output
        
        Parameters:
        inputs : array of shape (batch_size, num_features)
        training : bool - Whether in training mode
        
        Returns:
        array - Normalized, scaled, and shifted output
        """
        if training:
            # Compute batch statistics
            self.mean = np.mean(inputs, axis=0)
            self.var = np.var(inputs, axis=0)
            
            # Update running statistics using exponential moving average
            # running = momentum * running + (1-momentum) * batch
            self.running_mean = (
                self.momentum * self.running_mean + 
                (1 - self.momentum) * self.mean
            )
            self.running_var = (
                self.momentum * self.running_var + 
                (1 - self.momentum) * self.var
            )
            
            # Use batch statistics for normalization
            mean, var = self.mean, self.var
        else:
            # Use running statistics for inference
            mean, var = self.running_mean, self.running_var
        
        # Normalize: (x - μ) / √(σ² + ε)
        self.x_centered = inputs - mean
        self.std = np.sqrt(var + self.epsilon)
        self.x_norm = self.x_centered / self.std
        
        # Scale and shift: y = γ * x_norm + β
        output = self.gamma * self.x_norm + self.beta
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass: compute gradients w.r.t inputs and parameters.
        
        This is complex due to mean and variance depending on all batch samples.
        The gradient flows through normalization, mean, and variance computations.
        
        Gradients:
        - dL/dγ = sum(dL/dy * x_norm)
        - dL/dβ = sum(dL/dy)
        - dL/dx = ... (complex, involves chain rule through mean and var)
        
        Parameters:
        grad_output : array - Gradient w.r.t output
        
        Returns:
        array - Gradient w.r.t input
        """
        batch_size = grad_output.shape[0]
        
        # Gradient w.r.t scale parameter
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0)
        
        # Gradient w.r.t shift parameter
        self.grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t normalized input
        grad_x_norm = grad_output * self.gamma
        
        # Gradient w.r.t variance
        # d(x_norm)/d(var) = -0.5 * x_centered * (var + eps)^(-3/2)
        grad_var = np.sum(
            grad_x_norm * self.x_centered * -0.5 * (self.var + self.epsilon) ** (-1.5),
            axis=0
        )
        
        # Gradient w.r.t mean
        # Two paths: direct and through variance
        grad_mean = (
            np.sum(grad_x_norm * -1 / self.std, axis=0) +
            grad_var * np.mean(-2 * self.x_centered, axis=0)
        )
        
        # Gradient w.r.t input (three paths: through x_norm, var, and mean)
        grad_input = (
            grad_x_norm / self.std +
            grad_var * 2 * self.x_centered / batch_size +
            grad_mean / batch_size
        )
        
        return grad_input
    
    def get_params(self):
        """Return learnable parameters"""
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }
    
    def get_grads(self):
        """Return parameter gradients"""
        return {
            'gamma': self.grad_gamma,
            'beta': self.grad_beta
        }


class LayerNorm(Layer):
    """
    Layer Normalization: Normalize activations across feature dimension.
    
    Formula:
    - μ = mean(x, axis=-1)             # Feature mean per sample
    - σ² = var(x, axis=-1)             # Feature variance per sample
    - x_norm = (x - μ) / √(σ² + ε)    # Normalize
    - y = γ * x_norm + β               # Scale and shift
    
    Mathematical Properties:
    - Normalizes each sample independently across features
    - Does not depend on batch (unlike BatchNorm)
    - Same behavior during training and inference
    
    When to use:
    - Recurrent neural networks (RNNs, LSTMs, GRUs)
    - Transformers and attention mechanisms
    - Small batch sizes (BatchNorm struggles here)
    - Online learning (sample-by-sample)
    
    BatchNorm vs LayerNorm:
    - BatchNorm: normalize across batch for each feature
    - LayerNorm: normalize across features for each sample
    - BatchNorm: batch-dependent, different train/test behavior
    - LayerNorm: batch-independent, same train/test behavior
    
    Why it helps:
    - Stabilizes training in RNNs and Transformers
    - Reduces sensitivity to input scale
    - No dependence on batch size
    - Simpler inference (no running statistics needed)
    
    When to prefer over BatchNorm:
    - Sequential models (RNNs, Transformers)
    - Variable batch sizes
    - Online/streaming inference
    - When batch statistics don't make sense
    """
    
    def __init__(self, num_features, epsilon=1e-5):
        """
        Parameters:
        num_features : int - Number of features
        epsilon : float - Small constant for numerical stability
        """
        self.num_features = num_features
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(num_features)   # Scale
        self.beta = np.zeros(num_features)   # Shift
        
        # Cache for backward pass
        self.x_norm = None
        self.mean = None
        self.var = None
        self.x_centered = None
        self.std = None
        
        # Gradients
        self.grad_gamma = None
        self.grad_beta = None
    
    def forward(self, inputs, training=True):
        """
        Forward pass: normalize across features.
        
        Unlike BatchNorm:
        - Computes statistics per sample (across features)
        - Same behavior in training and inference
        - No running statistics needed
        
        Parameters:
        inputs : array of shape (batch_size, num_features)
        training : bool - Unused (same behavior for train/test)
        
        Returns:
        array - Normalized, scaled, and shifted output
        """
        # Compute mean and variance across features (axis=-1)
        # Keep dimensions for broadcasting
        self.mean = np.mean(inputs, axis=-1, keepdims=True)
        self.var = np.var(inputs, axis=-1, keepdims=True)
        
        # Normalize: (x - μ) / √(σ² + ε)
        self.x_centered = inputs - self.mean
        self.std = np.sqrt(self.var + self.epsilon)
        self.x_norm = self.x_centered / self.std
        
        # Scale and shift: y = γ * x_norm + β
        output = self.gamma * self.x_norm + self.beta
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass: compute gradients.
        
        Similar to BatchNorm but:
        - Statistics computed per sample, not per batch
        - Gradient flows across features for each sample
        
        Parameters:
        grad_output : array - Gradient w.r.t output
        
        Returns:
        array - Gradient w.r.t input
        """
        # Gradient w.r.t scale parameter
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0)
        
        # Gradient w.r.t shift parameter
        self.grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t normalized input
        grad_x_norm = grad_output * self.gamma
        
        # Gradient w.r.t variance (computed per sample across features)
        grad_var = np.sum(
            grad_x_norm * self.x_centered * -0.5 * (self.var + self.epsilon) ** (-1.5),
            axis=-1,
            keepdims=True
        )
        
        # Gradient w.r.t mean
        grad_mean = (
            np.sum(grad_x_norm * -1 / self.std, axis=-1, keepdims=True) +
            grad_var * np.mean(-2 * self.x_centered, axis=-1, keepdims=True)
        )
        
        # Gradient w.r.t input
        grad_input = (
            grad_x_norm / self.std +
            grad_var * 2 * self.x_centered / self.num_features +
            grad_mean / self.num_features
        )
        
        return grad_input
    
    def get_params(self):
        """Return learnable parameters"""
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }
    
    def get_grads(self):
        """Return parameter gradients"""
        return {
            'gamma': self.grad_gamma,
            'beta': self.grad_beta
        }
