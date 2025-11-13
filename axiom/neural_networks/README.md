# Neural Networks Module - README

## Overview

This module contains the fundamental building blocks for deep learning, implemented from first principles using only NumPy. Each component includes extensive mathematical explanations, intuitions, and best practices.

## Module Structure

```
neural_networks/
├── __init__.py           # Module exports
├── activations.py        # Activation functions
├── losses.py            # Loss functions
├── optimizers.py        # Optimization algorithms
├── layers.py            # Layer types
└── initializers.py      # Weight initialization
```

## Components

### 1. Activation Functions (`activations.py`)

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

**Available Activations:**

| Activation | Formula | Range | Use Case |
|-----------|---------|-------|----------|
| **Sigmoid** | σ(z) = 1/(1+e^(-z)) | (0, 1) | Binary classification output, LSTM gates |
| **Tanh** | tanh(z) | (-1, 1) | Hidden layers, RNNs |
| **ReLU** | max(0, z) | [0, ∞) | Default for hidden layers, CNNs |
| **LeakyReLU** | max(αz, z) | (-∞, ∞) | Prevents dying ReLU |
| **ELU** | z if z>0 else α(e^z-1) | (-α, ∞) | Alternative to ReLU with smooth negatives |
| **Softmax** | e^(z_i)/Σe^(z_j) | (0, 1), sum=1 | Multi-class classification output |

**Usage Example:**
```python
from axiom.neural_networks import ReLU, Sigmoid, get_activation

# Direct instantiation
relu = ReLU()
output = relu(inputs)

# Via string name
activation = get_activation('relu')
output = activation(inputs)

# Get gradient for backpropagation
gradient = relu.backward(inputs)
```

**When to use which:**
- **ReLU**: Default choice for hidden layers (fast, effective)
- **Sigmoid**: Binary classification output layer
- **Softmax**: Multi-class classification output layer
- **Tanh**: Hidden layers when zero-centered outputs preferred
- **LeakyReLU**: When dying ReLU is a problem

---

### 2. Loss Functions (`losses.py`)

Loss functions measure prediction error and guide the learning process.

**Available Losses:**

| Loss | Formula | Use Case |
|------|---------|----------|
| **MSE** | (1/n)Σ(y-ŷ)² | Regression |
| **MAE** | (1/n)Σ\|y-ŷ\| | Robust regression (outliers) |
| **Binary Cross-Entropy** | -Σ[y log(ŷ) + (1-y)log(1-ŷ)] | Binary classification |
| **Categorical Cross-Entropy** | -ΣΣ y_ij log(ŷ_ij) | Multi-class classification |
| **Hinge** | Σ max(0, 1-y*ŷ) | SVM, max-margin classification |
| **Huber** | MSE for small errors, MAE for large | Robust regression |

**Usage Example:**
```python
from axiom.neural_networks import MSELoss, CategoricalCrossEntropy

# Regression
loss_fn = MSELoss()
loss = loss_fn(y_true, y_pred)
gradient = loss_fn.backward(y_true, y_pred)

# Classification
loss_fn = CategoricalCrossEntropy()
loss = loss_fn(y_true_onehot, y_pred_probs)
```

**Pairing recommendations:**
- **Sigmoid + Binary Cross-Entropy**: Binary classification
- **Softmax + Categorical Cross-Entropy**: Multi-class classification
- **Linear + MSE**: Regression
- **ReLU + Huber**: Robust regression

---

### 3. Optimizers (`optimizers.py`)

Optimizers update model parameters to minimize loss.

**Available Optimizers:**

| Optimizer | Key Feature | Hyperparameters | Typical LR |
|-----------|-------------|-----------------|------------|
| **SGD** | Simple gradient descent | learning_rate | 0.01 - 0.1 |
| **SGD Momentum** | Accelerates convergence | learning_rate, momentum (0.9) | 0.01 - 0.1 |
| **RMSprop** | Adaptive per-parameter rates | learning_rate, decay_rate (0.9) | 0.001 |
| **Adam** | Combines momentum + RMSprop | learning_rate, β₁ (0.9), β₂ (0.999) | 0.001 |
| **AdaGrad** | Adapts to sparse data | learning_rate | 0.01 |

**Usage Example:**
```python
from axiom.neural_networks import Adam, SGD, get_optimizer

# Adam (default choice)
optimizer = Adam(learning_rate=0.001)

# Update parameters
params = {'weights': W, 'bias': b}
grads = {'weights': dW, 'bias': db}
updated_params = optimizer.update(params, grads)

# Via string name
optimizer = get_optimizer('adam', learning_rate=0.001)
```

**Which optimizer to choose:**
- **Adam**: Default choice, works well for most problems
- **SGD + Momentum**: When you want careful, controlled training
- **RMSprop**: Good for RNNs and non-stationary problems
- **AdaGrad**: Sparse data (NLP, recommendations)

**Quick comparison:**
```
Speed of convergence: Adam > RMSprop > SGD Momentum > SGD
Generalization: SGD ≈ SGD Momentum > Adam > RMSprop
Ease of use: Adam > RMSprop > SGD Momentum > SGD
```

---

### 4. Layers (`layers.py`)

Neural network layers transform inputs and contain learnable parameters.

**Available Layers:**

#### **Dense (Fully Connected)**
```python
from axiom.neural_networks import Dense

layer = Dense(input_dim=784, output_dim=128, use_bias=True)
layer.initialize('he')  # Initialize weights
output = layer(inputs, training=True)
```

**When to use:**
- Most common layer type
- After flattening CNN features
- Output layers
- Typical architecture: Input → Dense → Activation → Dense → ...

---

#### **Dropout**
```python
from axiom.neural_networks import Dropout

dropout = Dropout(dropout_rate=0.5)
output = dropout(inputs, training=True)  # Drops 50% during training
output = dropout(inputs, training=False)  # No dropout during inference
```

**When to use:**
- Prevent overfitting
- After dense or convolutional layers
- Typical rates: 0.5 for hidden layers, 0.2-0.3 for input

**Important:**
- Only active during training
- Must pass `training=False` during inference

---

#### **Batch Normalization**
```python
from axiom.neural_networks import BatchNorm

bn = BatchNorm(num_features=128, momentum=0.9)
output = bn(inputs, training=True)  # Uses batch statistics
output = bn(inputs, training=False)  # Uses running statistics
```

**When to use:**
- After linear/conv layers, before activation
- Deep networks (>5 layers)
- When you want faster training

**Benefits:**
- Faster convergence (can use higher learning rates)
- Less sensitive to initialization
- Implicit regularization

---

#### **Layer Normalization**
```python
from axiom.neural_networks import LayerNorm

ln = LayerNorm(num_features=128)
output = ln(inputs, training=True)
```

**When to use:**
- RNNs, LSTMs, Transformers
- Small batch sizes
- Online learning

**BatchNorm vs LayerNorm:**
- BatchNorm: normalizes across batch (each feature independently)
- LayerNorm: normalizes across features (each sample independently)
- BatchNorm: batch-dependent, different train/test behavior
- LayerNorm: batch-independent, same train/test behavior

---

### 5. Initializers (`initializers.py`)

Weight initialization strategies prevent vanishing/exploding gradients and accelerate training.

**Available Initializers:**

| Initializer | Formula | Use Case |
|------------|---------|----------|
| **Xavier Uniform** | U(-√(6/(n_in+n_out)), √(6/(n_in+n_out))) | Sigmoid, tanh |
| **Xavier Normal** | N(0, 2/(n_in+n_out)) | Sigmoid, tanh |
| **He Uniform** | U(-√(6/n_in), √(6/n_in)) | ReLU, LeakyReLU |
| **He Normal** | N(0, 2/n_in) | ReLU (default choice) |
| **LeCun** | Various | SELU activation |

**Usage Example:**
```python
from axiom.neural_networks import HeNormal, get_initializer
from axiom.neural_networks import get_recommended_initializer

# Direct instantiation
init = HeNormal()
weights = init((784, 128))

# Via string name
init = get_initializer('he')
weights = init((784, 128))

# Automatic based on activation
init = get_recommended_initializer('relu')  # Returns HeNormal
weights = init((784, 128))

# Initialize layer
layer = Dense(784, 128)
layer.initialize('he')
```

**Which initializer to choose:**

| Activation | Recommended Initializer | Why |
|-----------|------------------------|-----|
| ReLU, LeakyReLU, ELU | He Normal | Accounts for ReLU zeroing half the neurons |
| Sigmoid, Tanh | Xavier (Glorot) | Maintains variance across layers |
| SELU | LeCun | Preserves self-normalizing properties |

**Rule of thumb:**
```python
# Modern networks (with ReLU)
layer.initialize('he')

# Classic networks (with sigmoid/tanh)
layer.initialize('xavier')
```

---

## Complete Examples

### Example 1: Simple Binary Classification

```python
from axiom.neural_networks import Dense, Sigmoid, BinaryCrossEntropy, Adam

# Create layers
hidden = Dense(10, 64)
hidden.initialize('he')

output = Dense(64, 1)
output.initialize('xavier')

# Setup training
activation = Sigmoid()
loss_fn = BinaryCrossEntropy()
optimizer = Adam(learning_rate=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    h = hidden(X, training=True)
    a = activation(h)
    y_pred = output(a, training=True)
    y_pred = activation(y_pred)
    
    # Compute loss
    loss = loss_fn(y_true, y_pred)
    
    # Backward pass
    grad = loss_fn.backward(y_true, y_pred)
    grad = activation.backward(y_pred) * grad
    grad = output.backward(grad)
    # ... continue backprop
    
    # Update weights
    params = {**hidden.get_params(), **output.get_params()}
    grads = {**hidden.get_grads(), **output.get_grads()}
    updated = optimizer.update(params, grads)
```

### Example 2: Multi-class Classification with Regularization

```python
from axiom.neural_networks import (
    Dense, Dropout, BatchNorm, 
    ReLU, Softmax,
    CategoricalCrossEntropy, Adam
)

# Build network
hidden1 = Dense(784, 256)
hidden1.initialize('he')
bn1 = BatchNorm(256)
relu1 = ReLU()
dropout1 = Dropout(0.3)

hidden2 = Dense(256, 128)
hidden2.initialize('he')
bn2 = BatchNorm(128)
relu2 = ReLU()
dropout2 = Dropout(0.3)

output = Dense(128, 10)
output.initialize('xavier')
softmax = Softmax()

# Setup training
loss_fn = CategoricalCrossEntropy()
optimizer = Adam(learning_rate=0.001)

# Training
for epoch in range(epochs):
    # Forward (training mode)
    h1 = hidden1(X, training=True)
    h1 = bn1(h1, training=True)
    h1 = relu1(h1)
    h1 = dropout1(h1, training=True)
    
    h2 = hidden2(h1, training=True)
    h2 = bn2(h2, training=True)
    h2 = relu2(h2)
    h2 = dropout2(h2, training=True)
    
    y_pred = output(h2, training=True)
    y_pred = softmax(y_pred)
    
    # Loss and backward pass...
```

### Example 3: Regression with Robust Loss

```python
from axiom.neural_networks import Dense, ReLU, HuberLoss, Adam

# Build network
hidden = Dense(20, 64)
hidden.initialize('he')
relu = ReLU()

output = Dense(64, 1)
output.initialize('he')

# Huber loss for robustness to outliers
loss_fn = HuberLoss(delta=1.0)
optimizer = Adam(learning_rate=0.001)

# Training loop
for epoch in range(epochs):
    # Forward
    h = hidden(X, training=True)
    h = relu(h)
    y_pred = output(h, training=True)
    
    # Loss and optimization...
```

---

## Best Practices

### 1. Activation Function Choice
```python
# Hidden layers: ReLU (default)
hidden_activation = ReLU()

# Binary output: Sigmoid
output_activation = Sigmoid()

# Multi-class output: Softmax
output_activation = Softmax()

# If dying ReLU is a problem: LeakyReLU
hidden_activation = LeakyReLU(alpha=0.01)
```

### 2. Loss Function Pairing
```python
# Binary classification
loss = BinaryCrossEntropy()
output_activation = Sigmoid()

# Multi-class classification
loss = CategoricalCrossEntropy()
output_activation = Softmax()

# Regression
loss = MSELoss()
output_activation = None  # Linear

# Regression with outliers
loss = HuberLoss(delta=1.0)
output_activation = None
```

### 3. Initialization Strategy
```python
# Modern networks (ReLU)
layer.initialize('he')

# Classic networks (sigmoid/tanh)
layer.initialize('xavier')

# Automatic (recommended)
from axiom.neural_networks import get_recommended_initializer
init = get_recommended_initializer('relu')
```

### 4. Regularization
```python
# Dropout after activations
dropout = Dropout(0.5)  # 50% for hidden layers

# Batch normalization before activation
bn = BatchNorm(num_features)

# Typical pattern
x = Dense(n_in, n_out)(x)
x = BatchNorm(n_out)(x, training=training)
x = ReLU()(x)
x = Dropout(0.3)(x, training=training)
```

### 5. Training vs Inference
```python
# Training mode
output = dropout(x, training=True)   # Applies dropout
output = batchnorm(x, training=True)  # Uses batch statistics

# Inference mode
output = dropout(x, training=False)   # No dropout
output = batchnorm(x, training=False) # Uses running statistics
```

---

## Mathematical Background

### Why These Components?

1. **Activations**: Introduce non-linearity
   - Without: network is just linear transformation (y = Wx + b)
   - With: can approximate any function (universal approximation)

2. **Losses**: Define what "good" means
   - Classification: minimize prediction uncertainty (cross-entropy)
   - Regression: minimize prediction error (MSE)

3. **Optimizers**: Navigate loss landscape efficiently
   - Vanilla SGD: slow but stable
   - Adam: fast with adaptive learning rates

4. **Regularization** (Dropout, BatchNorm): Prevent overfitting
   - Dropout: ensemble of sub-networks
   - BatchNorm: normalizes activations, stabilizes training

5. **Initialization**: Start in right regime
   - Too small: vanishing gradients
   - Too large: exploding gradients
   - Just right: stable learning

---

## Common Patterns

### Standard Feedforward Block
```python
# Dense → BatchNorm → Activation → Dropout
x = Dense(n_in, n_out)(x)
x = BatchNorm(n_out)(x, training=training)
x = ReLU()(x)
x = Dropout(0.3)(x, training=training)
```

### Output Layer Patterns
```python
# Binary classification
x = Dense(n_hidden, 1)(x)
x = Sigmoid()(x)

# Multi-class classification
x = Dense(n_hidden, n_classes)(x)
x = Softmax()(x)

# Regression
x = Dense(n_hidden, n_outputs)(x)
# No activation (linear output)
```

---

## Tips for Success

1. **Start simple**: Use Adam optimizer, ReLU activation, He initialization
2. **Add complexity gradually**: Start without regularization, add if overfitting
3. **Monitor training**: Watch for vanishing/exploding gradients
4. **Use appropriate loss**: Match loss to problem type
5. **Remember training mode**: Dropout and BatchNorm behave differently in training vs inference

---

## References

- **Xavier Initialization**: Glorot & Bengio, 2010
- **He Initialization**: He et al., 2015
- **Batch Normalization**: Ioffe & Szegedy, 2015
- **Layer Normalization**: Ba et al., 2016
- **Adam Optimizer**: Kingma & Ba, 2014
- **Dropout**: Srivastava et al., 2014

---

## Next Steps

With these building blocks, you can:
1. Build feedforward neural networks
2. Implement backpropagation from scratch
3. Create custom architectures (CNNs, RNNs)
4. Experiment with different optimization strategies

See the examples in the main repository for complete implementations!
