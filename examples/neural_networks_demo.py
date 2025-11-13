# examples/neural_networks_demo.py
"""
Neural Network Building Blocks - Usage Examples
Demonstrates activation functions, losses, optimizers, layers, and initializers
"""

import numpy as np
import sys
sys.path.insert(0, '..')  # Allow imports when running from examples/

# Import all components with correct paths
from axiom.neural_networks import (
    # Activations
    ReLU, Sigmoid, Softmax, Tanh, LeakyReLU, get_activation,
    # Losses
    MSELoss, BinaryCrossEntropy, CategoricalCrossEntropy,
    # Optimizers
    Adam, SGD, RMSprop, get_optimizer,
    # Layers
    Dense, Dropout, BatchNorm, LayerNorm,
    # Initializers
    get_initializer, get_recommended_initializer
)


def example_1_simple_forward_pass():
    """
    Example 1: Simple forward pass through layers
    """
    print("=" * 60)
    print("Example 1: Simple Forward Pass")
    print("=" * 60)
    
    # Create dummy data
    X = np.random.randn(32, 10)  # 32 samples, 10 features
    
    # Create a simple network: Input(10) → Dense(64) → ReLU → Dense(1) → Sigmoid
    layer1 = Dense(10, 64)
    layer1.initialize('he')
    
    layer2 = Dense(64, 1)
    layer2.initialize('xavier')
    
    # Activations
    relu = ReLU()
    sigmoid = Sigmoid()
    
    # Forward pass
    h1 = layer1(X, training=True)
    a1 = relu(h1)
    h2 = layer2(a1, training=True)
    output = sigmoid(h2)
    
    print(f"Input shape: {X.shape}")
    print(f"After layer1: {h1.shape}")
    print(f"After ReLU: {a1.shape}")
    print(f"After layer2: {h2.shape}")
    print(f"Final output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print()


def example_2_loss_computation():
    """
    Example 2: Computing different loss functions
    """
    print("=" * 60)
    print("Example 2: Loss Computation")
    print("=" * 60)
    
    # Binary classification
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.95])
    
    bce = BinaryCrossEntropy()
    loss = bce(y_true, y_pred)
    gradient = bce.backward(y_true, y_pred)
    
    print("Binary Classification:")
    print(f"  True labels: {y_true}")
    print(f"  Predictions: {y_pred}")
    print(f"  BCE Loss: {loss:.4f}")
    print(f"  Gradient shape: {gradient.shape}")
    print()
    
    # Regression
    y_true_reg = np.array([1.5, 2.0, 3.5, 4.0])
    y_pred_reg = np.array([1.6, 1.9, 3.4, 4.2])
    
    mse = MSELoss()
    loss = mse(y_true_reg, y_pred_reg)
    gradient = mse.backward(y_true_reg, y_pred_reg)
    
    print("Regression:")
    print(f"  True values: {y_true_reg}")
    print(f"  Predictions: {y_pred_reg}")
    print(f"  MSE Loss: {loss:.4f}")
    print()


def example_3_optimizer_update():
    """
    Example 3: Using optimizers to update parameters
    """
    print("=" * 60)
    print("Example 3: Optimizer Updates")
    print("=" * 60)
    
    # Dummy parameters and gradients
    params = {
        'weights': np.random.randn(10, 5),
        'bias': np.random.randn(5)
    }
    
    grads = {
        'weights': np.random.randn(10, 5) * 0.1,
        'bias': np.random.randn(5) * 0.1
    }
    
    print("Comparing different optimizers:")
    print()
    
    # SGD
    sgd = SGD(learning_rate=0.01)
    updated_sgd = sgd.update(params.copy(), grads)
    weight_change_sgd = np.linalg.norm(updated_sgd['weights'] - params['weights'])
    print(f"SGD weight change magnitude: {weight_change_sgd:.6f}")
    
    # Adam
    adam = Adam(learning_rate=0.001)
    updated_adam = adam.update(params.copy(), grads)
    weight_change_adam = np.linalg.norm(updated_adam['weights'] - params['weights'])
    print(f"Adam weight change magnitude: {weight_change_adam:.6f}")
    print()


def example_4_dropout_batchnorm():
    """
    Example 4: Regularization with Dropout and BatchNorm
    """
    print("=" * 60)
    print("Example 4: Dropout and Batch Normalization")
    print("=" * 60)
    
    X = np.random.randn(32, 64)  # 32 samples, 64 features
    
    # Dropout
    dropout = Dropout(dropout_rate=0.5)
    
    # Training mode
    output_train = dropout(X, training=True)
    zeros_train = np.sum(output_train == 0) / output_train.size
    print(f"Dropout (training):")
    print(f"  Input mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"  Output mean: {output_train.mean():.4f}, std: {output_train.std():.4f}")
    print(f"  Percentage of zeros: {zeros_train * 100:.1f}%")
    print()
    
    # Inference mode
    output_test = dropout(X, training=False)
    zeros_test = np.sum(output_test == 0) / output_test.size
    print(f"Dropout (inference):")
    print(f"  Output mean: {output_test.mean():.4f}, std: {output_test.std():.4f}")
    print(f"  Percentage of zeros: {zeros_test * 100:.1f}%")
    print()
    
    # Batch Normalization
    bn = BatchNorm(num_features=64)
    
    output_bn = bn(X, training=True)
    print(f"Batch Normalization:")
    print(f"  Input mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"  Output mean: {output_bn.mean():.4f}, std: {output_bn.std():.4f}")
    print(f"  (Should be ~0 mean, ~1 std after normalization)")
    print()


def example_5_activation_functions():
    """
    Example 5: Testing different activation functions
    """
    print("=" * 60)
    print("Example 5: Activation Functions")
    print("=" * 60)
    
    # Test inputs
    x = np.array([-2, -1, 0, 1, 2])
    
    activations = {
        'ReLU': ReLU(),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'LeakyReLU': LeakyReLU()
    }
    
    for name, activation in activations.items():
        output = activation(x)
        gradient = activation.backward(x)
        print(f"{name}:")
        print(f"  Input:    {x}")
        print(f"  Output:   {output}")
        print(f"  Gradient: {gradient}")
        print()


def example_6_initialization_comparison():
    """
    Example 6: Comparing different initialization strategies
    """
    print("=" * 60)
    print("Example 6: Weight Initialization")
    print("=" * 60)
    
    shape = (100, 100)
    
    initializers = {
        'Xavier': 'xavier',
        'He': 'he',
        'LeCun': 'lecun_normal'
    }
    
    for name, init_name in initializers.items():
        init = get_initializer(init_name)
        weights = init(shape)
        print(f"{name} Initialization:")
        print(f"  Mean: {weights.mean():.6f}")
        print(f"  Std:  {weights.std():.6f}")
        print(f"  Min:  {weights.min():.6f}")
        print(f"  Max:  {weights.max():.6f}")
        print()
    
    # Automatic initialization based on activation
    print("Recommended initialization for ReLU:", 
          type(get_recommended_initializer('relu')).__name__)
    print("Recommended initialization for Sigmoid:", 
          type(get_recommended_initializer('sigmoid')).__name__)
    print()


def example_7_complete_training_step():
    """
    Example 7: Complete training step (forward + backward + update)
    """
    print("=" * 60)
    print("Example 7: Complete Training Step")
    print("=" * 60)
    
    # Create simple dataset
    np.random.seed(42)
    X = np.random.randn(16, 5)
    y = np.random.randint(0, 2, (16, 1))
    
    # Build network
    layer1 = Dense(5, 10)
    layer1.initialize('he')
    
    layer2 = Dense(10, 1)
    layer2.initialize('xavier')
    
    # Components
    relu = ReLU()
    sigmoid = Sigmoid()
    loss_fn = BinaryCrossEntropy()
    optimizer = Adam(learning_rate=0.01)
    
    print("Training for 5 steps...")
    for step in range(5):
        # Forward pass
        h1 = layer1(X, training=True)
        a1 = relu(h1)
        h2 = layer2(a1, training=True)
        y_pred = sigmoid(h2)
        
        # Compute loss
        loss = loss_fn(y, y_pred)
        
        # Backward pass
        grad = loss_fn.backward(y, y_pred)
        grad = sigmoid.backward(h2) * grad
        grad = layer2.backward(grad)
        grad = relu.backward(h1) * grad
        grad = layer1.backward(grad)
        
        # Update parameters
        params = {}
        grads = {}
        
        for i, layer in enumerate([layer1, layer2], 1):
            layer_params = layer.get_params()
            layer_grads = layer.get_grads()
            for key, val in layer_params.items():
                params[f'layer{i}_{key}'] = val
            for key, val in layer_grads.items():
                grads[f'layer{i}_{key}'] = val
        
        updated_params = optimizer.update(params, grads)
        
        # Apply updates
        layer1.weights = updated_params['layer1_weights']
        layer1.bias = updated_params['layer1_bias']
        layer2.weights = updated_params['layer2_weights']
        layer2.bias = updated_params['layer2_bias']
        
        print(f"  Step {step + 1}: Loss = {loss:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NEURAL NETWORK BUILDING BLOCKS - USAGE EXAMPLES")
    print("=" * 60 + "\n")
    
    example_1_simple_forward_pass()
    example_2_loss_computation()
    example_3_optimizer_update()
    example_4_dropout_batchnorm()
    example_5_activation_functions()
    example_6_initialization_comparison()
    example_7_complete_training_step()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
