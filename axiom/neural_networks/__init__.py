# axiom/neural_networks/__init__.py
"""
Neural Networks Module - Building Deep Learning from First Principles

This module contains the fundamental building blocks for constructing
neural networks from scratch using only NumPy.

Components:
-----------
activations : Activation functions (ReLU, Sigmoid, Tanh, etc.)
losses : Loss functions (MSE, Cross-Entropy, Hinge, etc.)
optimizers : Optimization algorithms (SGD, Adam, RMSprop, etc.)
layers : Layer types (Dense, Dropout, BatchNorm, LayerNorm)
initializers : Weight initialization strategies (Xavier, He, etc.)

Philosophy:
-----------
Each component is implemented with:
- Clear mathematical explanations
- Detailed comments on the "why" behind design choices
- Educational focus over performance
- NumPy-only implementation for transparency

Example Usage:
--------------
>>> from axiom.neural_networks import Dense, ReLU, Adam, MSELoss
>>> from axiom.neural_networks import XavierUniform
>>> 
>>> # Create a dense layer with 784 inputs, 128 outputs
>>> layer = Dense(784, 128)
>>> layer.initialize('he')  # He initialization for ReLU
>>> 
>>> # Use activation
>>> relu = ReLU()
>>> output = relu(layer(inputs))
>>> 
>>> # Setup optimizer and loss
>>> optimizer = Adam(learning_rate=0.001)
>>> loss_fn = MSELoss()
"""

from .activations import (
    Activation,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Softmax,
    ELU,
    get_activation,
    ACTIVATIONS
)

from .losses import (
    Loss,
    MSELoss,
    MAELoss,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HingeLoss,
    HuberLoss,
    get_loss,
    LOSSES
)

from .optimizers import (
    Optimizer,
    SGD,
    SGDMomentum,
    RMSprop,
    Adam,
    AdaGrad,
    get_optimizer,
    OPTIMIZERS
)

from .layers import (
    Layer,
    Dense,
    Dropout,
    BatchNorm,
    LayerNorm
)

from .initializers import (
    Initializer,
    XavierUniform,
    XavierNormal,
    HeUniform,
    HeNormal,
    Uniform,
    Normal,
    Zeros,
    Ones,
    LecunUniform,
    LecunNormal,
    get_initializer,
    get_recommended_initializer,
    INITIALIZERS,
    RECOMMENDED_INITIALIZERS
)

__all__ = [
    # Activations
    'Activation',
    'Sigmoid',
    'Tanh',
    'ReLU',
    'LeakyReLU',
    'Softmax',
    'ELU',
    'get_activation',
    'ACTIVATIONS',
    
    # Losses
    'Loss',
    'MSELoss',
    'MAELoss',
    'BinaryCrossEntropy',
    'CategoricalCrossEntropy',
    'HingeLoss',
    'HuberLoss',
    'get_loss',
    'LOSSES',
    
    # Optimizers
    'Optimizer',
    'SGD',
    'SGDMomentum',
    'RMSprop',
    'Adam',
    'AdaGrad',
    'get_optimizer',
    'OPTIMIZERS',
    
    # Layers
    'Layer',
    'Dense',
    'Dropout',
    'BatchNorm',
    'LayerNorm',
    
    # Initializers
    'Initializer',
    'XavierUniform',
    'XavierNormal',
    'HeUniform',
    'HeNormal',
    'Uniform',
    'Normal',
    'Zeros',
    'Ones',
    'LecunUniform',
    'LecunNormal',
    'get_initializer',
    'get_recommended_initializer',
    'INITIALIZERS',
    'RECOMMENDED_INITIALIZERS'
]
