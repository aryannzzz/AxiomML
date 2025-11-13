# axiom/neural_networks/initializers.py
import numpy as np

class Initializer:
    """
    Base class for weight initialization strategies.
    Proper initialization is crucial for:
    - Preventing vanishing/exploding gradients
    - Faster convergence
    - Better final performance
    """
    
    def __call__(self, shape):
        """Generate initialized weights of given shape"""
        raise NotImplementedError


class XavierUniform(Initializer):
    """
    Xavier (Glorot) Uniform Initialization
    
    Formula: W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
    
    Mathematical Properties:
    - Draws weights from uniform distribution
    - Scale based on fan-in and fan-out
    - Designed to keep variance constant across layers
    
    When to use:
    - Sigmoid or tanh activation functions
    - Symmetric activations (output centered at 0)
    - Default choice for many applications
    
    Why it works:
    - Maintains variance of activations forward
    - Maintains variance of gradients backward
    - Prevents vanishing/exploding gradients
    
    Mathematical intuition:
    - With n inputs, variance of sum is n * var(weight)
    - To keep output variance ~1, need var(weight) ~ 1/n
    - Xavier accounts for both input and output dimensions
    - Uniform(-a, a) has variance a²/3, hence the √6 factor
    
    Derivation:
    - Want: Var(output) ≈ Var(input)
    - For linear layer: output = Σ(weight_i * input_i)
    - Var(output) = n_in * Var(weight) * Var(input)
    - Set Var(weight) = 1/n_in for forward pass
    - Similar analysis for backward gives 1/n_out
    - Compromise: use 2/(n_in + n_out)
    - For uniform distribution: scale = √(6 / (n_in + n_out))
    
    Original paper:
    - "Understanding the difficulty of training deep feedforward neural networks"
    - Glorot & Bengio, 2010
    """
    
    def __call__(self, shape):
        """
        Generate Xavier uniform initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix (typically (n_in, n_out))
        
        Returns:
        array - Initialized weights
        """
        if len(shape) < 2:
            # For bias or 1D parameters, just use small random values
            return np.random.uniform(-0.01, 0.01, shape)
        
        # Calculate fan-in and fan-out
        # fan_in: number of inputs to each neuron
        # fan_out: number of outputs from layer
        fan_in, fan_out = shape[0], shape[1]
        
        # Xavier scale factor
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        # Sample from uniform distribution
        return np.random.uniform(-limit, limit, shape)


class XavierNormal(Initializer):
    """
    Xavier (Glorot) Normal Initialization
    
    Formula: W ~ N(0, 2/(n_in + n_out))
    
    Mathematical Properties:
    - Draws weights from normal (Gaussian) distribution
    - Mean = 0, Variance = 2/(n_in + n_out)
    - Normal distribution alternative to Xavier Uniform
    
    When to use:
    - Same as Xavier Uniform
    - When you prefer normal over uniform distribution
    - Slightly more values near zero than Xavier Uniform
    
    Uniform vs Normal:
    - Uniform: bounded, flat distribution within range
    - Normal: unbounded, more values near mean (zero)
    - Empirically, both work well (slight preference for uniform)
    - Normal is more "natural" but uniform has theoretical justification
    
    Why normal distribution?:
    - Central limit theorem: sums tend toward normal
    - Easy to control variance directly
    - More values near zero (stronger regularization effect)
    - Unbounded (very rare extreme values possible)
    """
    
    def __call__(self, shape):
        """
        Generate Xavier normal initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - Initialized weights
        """
        if len(shape) < 2:
            return np.random.normal(0, 0.01, shape)
        
        fan_in, fan_out = shape[0], shape[1]
        
        # Standard deviation for normal distribution
        std = np.sqrt(2.0 / (fan_in + fan_out))
        
        # Sample from normal distribution
        return np.random.normal(0, std, shape)


class HeUniform(Initializer):
    """
    He Uniform Initialization
    
    Formula: W ~ U(-√(6/n_in), √(6/n_in))
    
    Mathematical Properties:
    - Specifically designed for ReLU activation
    - Only considers fan-in (input dimension)
    - Larger variance than Xavier
    
    When to use:
    - ReLU, LeakyReLU, or other ReLU-like activations
    - Deep networks with ReLU
    - Modern CNNs and ResNets
    
    Why different from Xavier?:
    - ReLU kills half the neurons (outputs 0 for negative inputs)
    - This halves the variance during forward pass
    - He initialization compensates by doubling the variance
    - Derived specifically for ReLU's properties
    
    Mathematical intuition:
    - ReLU(x) = max(0, x) zeroes out negative values
    - Reduces variance by approximately half
    - Need higher initial variance to compensate
    - Formula: Var(weight) = 2/n_in (twice Xavier)
    
    Why only fan-in?:
    - ReLU's asymmetry breaks Xavier's symmetry assumption
    - Forward pass variance more critical than backward
    - Empirically works better for ReLU than Xavier
    
    Original paper:
    - "Delving Deep into Rectifiers: Surpassing Human-Level Performance"
    - He et al., 2015
    """
    
    def __call__(self, shape):
        """
        Generate He uniform initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - Initialized weights
        """
        if len(shape) < 2:
            return np.random.uniform(-0.01, 0.01, shape)
        
        # Only use fan-in for He initialization
        fan_in = shape[0]
        
        # He scale factor (note: only fan_in, not fan_in + fan_out)
        limit = np.sqrt(6.0 / fan_in)
        
        return np.random.uniform(-limit, limit, shape)


class HeNormal(Initializer):
    """
    He Normal Initialization
    
    Formula: W ~ N(0, 2/n_in)
    
    Mathematical Properties:
    - Normal distribution variant of He initialization
    - Designed for ReLU activations
    - Most commonly used initializer in modern deep learning
    
    When to use:
    - Default choice for ReLU networks
    - CNNs, ResNets, modern architectures
    - When using ReLU, LeakyReLU, PReLU
    
    Why it's the default:
    - ReLU is default activation in modern networks
    - He initialization specifically designed for ReLU
    - Normal distribution is standard choice
    - Empirically proven across many architectures
    
    PyTorch and TensorFlow default:
    - PyTorch: uses He normal (called Kaiming normal)
    - TensorFlow: also defaults to He normal for many layers
    - Industry standard for initializing deep networks
    """
    
    def __call__(self, shape):
        """
        Generate He normal initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - Initialized weights
        """
        if len(shape) < 2:
            return np.random.normal(0, 0.01, shape)
        
        fan_in = shape[0]
        
        # Standard deviation for He normal
        std = np.sqrt(2.0 / fan_in)
        
        return np.random.normal(0, std, shape)


class Uniform(Initializer):
    """
    Simple Uniform Initialization
    
    Formula: W ~ U(-scale, scale)
    
    When to use:
    - When you want explicit control over initialization range
    - Embedding layers
    - Custom applications
    
    Properties:
    - Simple, interpretable
    - Bounded values
    - No automatic scaling
    
    Note: Less principled than Xavier/He, but sometimes useful.
    """
    
    def __init__(self, scale=0.05):
        """
        Parameters:
        scale : float - Range of uniform distribution is [-scale, scale]
        """
        self.scale = scale
    
    def __call__(self, shape):
        """
        Generate uniformly initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - Initialized weights from U(-scale, scale)
        """
        return np.random.uniform(-self.scale, self.scale, shape)


class Normal(Initializer):
    """
    Simple Normal (Gaussian) Initialization
    
    Formula: W ~ N(mean, std²)
    
    When to use:
    - When you want explicit control over distribution
    - Custom applications
    - Baseline comparisons
    
    Properties:
    - Unbounded (rare extreme values possible)
    - More values near mean
    - No automatic scaling
    
    Note: Less principled than Xavier/He, but sometimes useful.
    """
    
    def __init__(self, mean=0.0, std=0.05):
        """
        Parameters:
        mean : float - Mean of normal distribution
        std : float - Standard deviation
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, shape):
        """
        Generate normally initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - Initialized weights from N(mean, std²)
        """
        return np.random.normal(self.mean, self.std, shape)


class Zeros(Initializer):
    """
    Zero Initialization: W = 0
    
    When to use:
    - Bias initialization (common default)
    - BatchNorm shift parameter (β)
    - Some specialized cases
    
    Warning for weights:
    - NEVER use for weight matrices in hidden layers
    - All neurons would compute identical outputs
    - Gradients would be identical (symmetry problem)
    - Network cannot learn (weights stay identical)
    
    Why zeros for biases is OK:
    - Biases don't have symmetry problem
    - Weights break symmetry, biases just shift
    - Common default for bias initialization
    """
    
    def __call__(self, shape):
        """
        Generate zero-initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - All zeros
        """
        return np.zeros(shape)


class Ones(Initializer):
    """
    Ones Initialization: W = 1
    
    When to use:
    - BatchNorm scale parameter (γ)
    - LayerNorm scale parameter
    - Some gating mechanisms
    
    Warning for weights:
    - NEVER use for weight matrices in hidden layers
    - Same symmetry problem as zeros
    - All neurons would have identical gradients
    """
    
    def __call__(self, shape):
        """
        Generate ones-initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - All ones
        """
        return np.ones(shape)


class LecunUniform(Initializer):
    """
    LeCun Uniform Initialization
    
    Formula: W ~ U(-√(3/n_in), √(3/n_in))
    
    Mathematical Properties:
    - Precursor to Xavier initialization
    - Only uses fan-in
    - Designed for linear/tanh activations
    
    When to use:
    - SELU activation (self-normalizing networks)
    - Historical interest
    - Rarely used now (Xavier/He are better)
    
    Historical note:
    - Introduced by Yann LeCun in the 1990s
    - One of the first principled initialization methods
    - Paved the way for Xavier and He
    """
    
    def __call__(self, shape):
        """
        Generate LeCun uniform initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - Initialized weights
        """
        if len(shape) < 2:
            return np.random.uniform(-0.01, 0.01, shape)
        
        fan_in = shape[0]
        limit = np.sqrt(3.0 / fan_in)
        
        return np.random.uniform(-limit, limit, shape)


class LecunNormal(Initializer):
    """
    LeCun Normal Initialization
    
    Formula: W ~ N(0, 1/n_in)
    
    When to use:
    - SELU activation (self-normalizing networks)
    - Historical comparisons
    
    SELU networks:
    - SELU: Scaled Exponential Linear Unit
    - Has self-normalizing properties
    - LeCun initialization maintains these properties
    """
    
    def __call__(self, shape):
        """
        Generate LeCun normal initialized weights.
        
        Parameters:
        shape : tuple - Shape of weight matrix
        
        Returns:
        array - Initialized weights
        """
        if len(shape) < 2:
            return np.random.normal(0, 0.01, shape)
        
        fan_in = shape[0]
        std = np.sqrt(1.0 / fan_in)
        
        return np.random.normal(0, std, shape)


# Convenience dictionary for easy initializer lookup
INITIALIZERS = {
    'xavier': XavierUniform,
    'xavier_uniform': XavierUniform,
    'glorot': XavierUniform,
    'glorot_uniform': XavierUniform,
    'xavier_normal': XavierNormal,
    'glorot_normal': XavierNormal,
    'he': HeNormal,
    'he_normal': HeNormal,
    'kaiming_normal': HeNormal,
    'he_uniform': HeUniform,
    'kaiming_uniform': HeUniform,
    'uniform': Uniform,
    'normal': Normal,
    'zeros': Zeros,
    'zero': Zeros,
    'ones': Ones,
    'one': Ones,
    'lecun_uniform': LecunUniform,
    'lecun_normal': LecunNormal
}


def get_initializer(name):
    """
    Get initializer by name.
    
    Parameters:
    name : str or Initializer - Name of initializer or initializer instance
    
    Returns:
    Initializer instance
    
    Example:
    >>> init = get_initializer('xavier')
    >>> weights = init((784, 128))
    
    >>> init = get_initializer('he')
    >>> weights = init((128, 64))
    
    >>> init = get_initializer(HeNormal())
    >>> weights = init((64, 10))
    """
    if isinstance(name, Initializer):
        return name
    
    if isinstance(name, str):
        name = name.lower()
        if name in INITIALIZERS:
            return INITIALIZERS[name]()
        else:
            raise ValueError(
                f"Unknown initializer: {name}. "
                f"Choose from {list(INITIALIZERS.keys())}"
            )
    
    raise TypeError(f"Initializer must be string or Initializer instance, got {type(name)}")


# Recommended initializers by activation function
RECOMMENDED_INITIALIZERS = {
    'relu': 'he_normal',
    'leaky_relu': 'he_normal',
    'leakyrelu': 'he_normal',
    'elu': 'he_normal',
    'selu': 'lecun_normal',
    'sigmoid': 'xavier',
    'tanh': 'xavier',
    'softmax': 'xavier',
    'linear': 'xavier'
}


def get_recommended_initializer(activation):
    """
    Get recommended initializer for an activation function.
    
    Rules of thumb:
    - ReLU family → He initialization
    - Sigmoid/tanh → Xavier initialization
    - SELU → LeCun initialization
    
    Parameters:
    activation : str - Name of activation function
    
    Returns:
    Initializer instance
    
    Example:
    >>> init = get_recommended_initializer('relu')
    >>> # Returns HeNormal initializer
    """
    activation = activation.lower() if isinstance(activation, str) else 'relu'
    
    if activation in RECOMMENDED_INITIALIZERS:
        return get_initializer(RECOMMENDED_INITIALIZERS[activation])
    else:
        # Default to He for unknown activations (most are ReLU-like)
        return get_initializer('he_normal')
