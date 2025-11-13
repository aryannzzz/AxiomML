# Neural Networks Building Blocks - Quick Reference

## 🎯 One-Page Cheat Sheet

### Import Everything
```python
from axiom.neural_networks import *
```

---

## 🔥 Activations

| Name | Code | Output Range | Use Case |
|------|------|--------------|----------|
| **ReLU** | `ReLU()` | [0, ∞) | Hidden layers (default) |
| **Sigmoid** | `Sigmoid()` | (0, 1) | Binary output |
| **Softmax** | `Softmax()` | (0, 1), Σ=1 | Multi-class output |
| **Tanh** | `Tanh()` | (-1, 1) | Hidden layers (alternative) |
| **LeakyReLU** | `LeakyReLU(alpha=0.01)` | (-∞, ∞) | Fixes dying ReLU |

**Quick Pick:**
- Hidden layers → `ReLU()`
- Binary classification → `Sigmoid()`
- Multi-class → `Softmax()`

---

## 📉 Losses

| Name | Code | Task |
|------|------|------|
| **MSE** | `MSELoss()` | Regression |
| **BCE** | `BinaryCrossEntropy()` | Binary classification |
| **CCE** | `CategoricalCrossEntropy()` | Multi-class classification |
| **Huber** | `HuberLoss(delta=1.0)` | Robust regression |
| **Hinge** | `HingeLoss()` | SVM classification |

**Standard Pairings:**
```python
# Binary classification
Sigmoid() + BinaryCrossEntropy()

# Multi-class
Softmax() + CategoricalCrossEntropy()

# Regression
None + MSELoss()
```

---

## 🎯 Optimizers

| Name | Code | When to Use |
|------|------|-------------|
| **Adam** | `Adam(lr=0.001)` | Default choice (99% of cases) |
| **SGD** | `SGD(lr=0.01)` | Baseline / careful training |
| **SGD+Momentum** | `SGDMomentum(lr=0.01, momentum=0.9)` | Better than vanilla SGD |
| **RMSprop** | `RMSprop(lr=0.001)` | RNNs, non-stationary |
| **AdaGrad** | `AdaGrad(lr=0.01)` | Sparse data |

**Default choice:** `Adam(learning_rate=0.001)`

**Typical learning rates:**
- Adam/RMSprop: 0.001
- SGD: 0.01 - 0.1

---

## 🧱 Layers

### Dense (Fully Connected)
```python
layer = Dense(input_dim=784, output_dim=128)
layer.initialize('he')  # or 'xavier'
output = layer(input, training=True)
```

### Dropout (Regularization)
```python
dropout = Dropout(dropout_rate=0.5)
output = dropout(input, training=True)  # 50% dropped in training
output = dropout(input, training=False) # No dropout in inference
```

### Batch Normalization
```python
bn = BatchNorm(num_features=128)
output = bn(input, training=True)  # Normalize using batch stats
```

### Layer Normalization
```python
ln = LayerNorm(num_features=128)
output = ln(input, training=True)  # Normalize per sample
```

---

## 🎲 Initialization

| Method | Code | For Activation |
|--------|------|----------------|
| **He** | `'he'` or `'he_normal'` | ReLU, LeakyReLU (default) |
| **Xavier** | `'xavier'` or `'glorot'` | Sigmoid, Tanh |
| **LeCun** | `'lecun_normal'` | SELU |

```python
# Initialize layer
layer.initialize('he')  # Recommended for ReLU

# Or get initializer directly
init = get_initializer('he')
weights = init((784, 128))
```

**Auto-select based on activation:**
```python
init = get_recommended_initializer('relu')  # Returns He
init = get_recommended_initializer('sigmoid')  # Returns Xavier
```

---

## 🏗️ Standard Network Patterns

### Binary Classification
```python
# Network: Input → Dense → ReLU → Dense → Sigmoid
hidden = Dense(input_dim, 64)
hidden.initialize('he')
output = Dense(64, 1)
output.initialize('xavier')

# Training components
relu = ReLU()
sigmoid = Sigmoid()
loss_fn = BinaryCrossEntropy()
optimizer = Adam(learning_rate=0.001)
```

### Multi-class Classification
```python
# Network: Input → Dense → ReLU → Dense → Softmax
hidden = Dense(input_dim, 128)
hidden.initialize('he')
output = Dense(128, num_classes)
output.initialize('xavier')

# Training components
relu = ReLU()
softmax = Softmax()
loss_fn = CategoricalCrossEntropy()
optimizer = Adam(learning_rate=0.001)
```

### Regression
```python
# Network: Input → Dense → ReLU → Dense
hidden = Dense(input_dim, 64)
hidden.initialize('he')
output = Dense(64, output_dim)
output.initialize('he')

# Training components
relu = ReLU()
loss_fn = MSELoss()  # or HuberLoss() for outliers
optimizer = Adam(learning_rate=0.001)
```

### Deep Network with Regularization
```python
# Pattern: Dense → BatchNorm → ReLU → Dropout
layer1 = Dense(input_dim, 256)
layer1.initialize('he')
bn1 = BatchNorm(256)
relu = ReLU()
dropout1 = Dropout(0.3)

layer2 = Dense(256, 128)
layer2.initialize('he')
bn2 = BatchNorm(128)
dropout2 = Dropout(0.3)

output = Dense(128, num_classes)
output.initialize('xavier')
softmax = Softmax()

# Training
loss_fn = CategoricalCrossEntropy()
optimizer = Adam(learning_rate=0.001)
```

---

## 🔄 Training Loop Template

```python
# Setup
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossEntropy()

for epoch in range(num_epochs):
    # Forward pass
    h1 = layer1(X, training=True)
    a1 = activation1(h1)
    h2 = layer2(a1, training=True)
    y_pred = activation2(h2)
    
    # Compute loss
    loss = loss_fn(y_true, y_pred)
    
    # Backward pass
    grad = loss_fn.backward(y_true, y_pred)
    grad = activation2.backward(h2) * grad
    grad = layer2.backward(grad)
    grad = activation1.backward(h1) * grad
    grad = layer1.backward(grad)
    
    # Update parameters
    params = {**layer1.get_params(), **layer2.get_params()}
    grads = {**layer1.get_grads(), **layer2.get_grads()}
    updated = optimizer.update(params, grads)
    
    # Apply updates...
```

---

## 💡 Quick Tips

### ✅ Do's
- Use `Adam` optimizer (default choice)
- Use `ReLU` for hidden layers
- Initialize with `'he'` for ReLU
- Use `BatchNorm` for deep networks
- Use `Dropout(0.3-0.5)` to prevent overfitting
- Pass `training=False` during inference

### ❌ Don'ts
- Don't use sigmoid for hidden layers
- Don't forget to set `training=False` for Dropout/BatchNorm
- Don't use MSE for classification
- Don't use cross-entropy for regression
- Don't forget to initialize weights

---

## 📊 Debugging Checklist

**Loss not decreasing?**
- ✓ Check learning rate (too high or too low?)
- ✓ Verify loss function matches task
- ✓ Check if gradients are computed correctly
- ✓ Ensure proper initialization

**Overfitting?**
- ✓ Add Dropout
- ✓ Add BatchNorm
- ✓ Reduce model size
- ✓ Get more training data

**Underfitting?**
- ✓ Increase model size
- ✓ Train longer
- ✓ Remove too much regularization
- ✓ Check for bugs in forward pass

**NaN or Inf values?**
- ✓ Reduce learning rate
- ✓ Check weight initialization
- ✓ Add gradient clipping
- ✓ Check for numerical instability

---

## 🚀 Example: Complete Classifier

```python
from axiom.neural_networks import *
import numpy as np

# Data
X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 2, (1000, 1))

# Build network
hidden = Dense(20, 64)
hidden.initialize('he')
output = Dense(64, 1)
output.initialize('xavier')

# Components
relu = ReLU()
sigmoid = Sigmoid()
loss_fn = BinaryCrossEntropy()
optimizer = Adam(learning_rate=0.001)

# Train
for epoch in range(100):
    # Forward
    h = hidden(X_train, training=True)
    a = relu(h)
    y_pred = output(a, training=True)
    y_pred = sigmoid(y_pred)
    
    # Loss
    loss = loss_fn(y_train, y_pred)
    
    # Backward
    grad = loss_fn.backward(y_train, y_pred)
    grad = sigmoid.backward(output.inputs @ output.weights + output.bias) * grad
    grad = output.backward(grad)
    grad = relu.backward(h) * grad
    grad = hidden.backward(grad)
    
    # Update
    params = {**hidden.get_params(), **output.get_params()}
    grads = {**hidden.get_grads(), **output.get_grads()}
    updated = optimizer.update(params, grads)
    
    # Apply
    hidden.weights = updated['hidden_weights']
    output.weights = updated['output_weights']
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

---

## 📚 More Info

- See `README.md` for detailed documentation
- See `example_usage.py` for 7 complete examples
- See `INSTALLATION.md` for setup instructions

**Created for AxiomML - Building ML from First Principles** 🚀
