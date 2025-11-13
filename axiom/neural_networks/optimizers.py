# axiom/neural_networks/optimizers.py
import numpy as np

class Optimizer:
    """
    Base class for optimization algorithms.
    Optimizers update model parameters to minimize the loss function.
    They determine HOW we traverse the loss landscape during training.
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Parameters:
        learning_rate : float - Step size for parameter updates
        """
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        Update parameters based on gradients.
        
        Parameters:
        params : dict - Model parameters
        grads : dict - Gradients of loss w.r.t parameters
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset optimizer state (for momentum, velocity, etc.)"""
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    
    Update rule: θ = θ - α * ∇L(θ)
    
    Mathematical Properties:
    - Simplest optimization algorithm
    - Moves parameters in direction opposite to gradient
    - Step size controlled by learning rate α
    
    When to use:
    - Baseline optimizer for comparison
    - When dataset is small and smooth
    - When you want simple, interpretable training
    
    Advantages:
    - Simple to understand and implement
    - Computationally efficient
    - Well-studied theoretical properties
    
    Drawbacks:
    - Can be slow to converge
    - Sensitive to learning rate choice
    - Struggles with saddle points and ravines
    - Same learning rate for all parameters
    
    Why "Stochastic"?:
    - Originally, uses random mini-batches instead of full dataset
    - Each mini-batch gives noisy estimate of true gradient
    - This noise can help escape local minima
    - Allows training on datasets too large for memory
    
    Intuition:
    - Imagine rolling a ball down a hill
    - Gradient tells us the steepest downward direction
    - Learning rate controls how far we step
    - Simple but effective for many problems
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Parameters:
        learning_rate : float - Step size (typical range: 0.001 to 0.1)
        """
        super().__init__(learning_rate)
    
    def update(self, params, grads):
        """
        Update parameters using vanilla gradient descent.
        
        Formula: θ_new = θ_old - α * ∇L
        
        Parameters:
        params : dict - {name: parameter_array}
        grads : dict - {name: gradient_array}
        """
        updated_params = {}
        
        for name in params:
            if name in grads:
                # Simple gradient descent update
                updated_params[name] = params[name] - self.learning_rate * grads[name]
            else:
                # Keep parameter unchanged if no gradient
                updated_params[name] = params[name]
        
        return updated_params


class SGDMomentum(Optimizer):
    """
    SGD with Momentum
    
    Update rule:
    - v = β * v + ∇L(θ)
    - θ = θ - α * v
    
    Mathematical Properties:
    - Accumulates exponentially decayed moving average of gradients
    - Accelerates in directions of consistent gradient
    - Dampens oscillations in other directions
    
    When to use:
    - When loss surface has ravines (steep in some directions, shallow in others)
    - To accelerate convergence over vanilla SGD
    - Most deep learning problems (better than vanilla SGD)
    
    Advantages:
    - Faster convergence than SGD
    - Reduces oscillations
    - Can escape shallow local minima
    - Smoother optimization trajectory
    
    Why momentum helps:
    - Like a ball rolling downhill gaining speed
    - Past gradients influence current direction
    - β controls how much history to retain (typically 0.9)
    - Builds velocity in consistent directions
    
    Intuition:
    - Without momentum: cautious steps in gradient direction
    - With momentum: builds speed if gradients agree
    - Dampens zigzagging in narrow valleys
    - Think of a heavy ball with inertia
    
    Typical hyperparameters:
    - learning_rate: 0.01 to 0.1
    - momentum: 0.9 (retains 90% of previous velocity)
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Parameters:
        learning_rate : float - Step size
        momentum : float - Decay rate for velocity (typically 0.9)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        """
        Update parameters using momentum.
        
        Formula:
        - v_new = β * v_old + ∇L
        - θ_new = θ_old - α * v_new
        
        The velocity accumulates gradients with exponential decay.
        """
        updated_params = {}
        
        for name in params:
            if name in grads:
                # Initialize velocity for this parameter if needed
                if name not in self.velocity:
                    self.velocity[name] = np.zeros_like(params[name])
                
                # Update velocity: exponentially weighted average
                self.velocity[name] = (
                    self.momentum * self.velocity[name] + grads[name]
                )
                
                # Update parameters using velocity
                updated_params[name] = params[name] - self.learning_rate * self.velocity[name]
            else:
                updated_params[name] = params[name]
        
        return updated_params
    
    def reset(self):
        """Reset velocity to zero"""
        self.velocity = {}


class RMSprop(Optimizer):
    """
    Root Mean Square Propagation
    
    Update rule:
    - s = β * s + (1-β) * ∇L²
    - θ = θ - α * ∇L / (√s + ε)
    
    Mathematical Properties:
    - Adapts learning rate for each parameter
    - Divides by running average of gradient magnitudes
    - Automatically adjusts step size based on gradient history
    
    When to use:
    - Recurrent neural networks (where it was invented)
    - When parameters have very different scales
    - Non-stationary objectives (loss landscape changes)
    
    Advantages:
    - Adaptive learning rates per parameter
    - Works well with mini-batches
    - Less sensitive to learning rate choice
    - Good for non-stationary problems
    
    Why it works:
    - Parameters with large gradients get smaller effective learning rates
    - Parameters with small gradients get larger effective learning rates
    - Balances progress across all parameters
    - Prevents any single parameter from dominating
    
    Intuition:
    - Large, consistent gradients → smaller steps (already learning well)
    - Small, noisy gradients → larger steps (need more exploration)
    - Automatically calibrates learning for each dimension
    
    Typical hyperparameters:
    - learning_rate: 0.001 (often lower than SGD)
    - decay_rate: 0.9 (controls how much history to use)
    - epsilon: 1e-8 (prevents division by zero)
    """
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        """
        Parameters:
        learning_rate : float - Base learning rate
        decay_rate : float - Decay rate for squared gradients (β)
        epsilon : float - Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.squared_grads = {}
    
    def update(self, params, grads):
        """
        Update parameters using RMSprop.
        
        Formula:
        - s_new = β * s_old + (1-β) * ∇L²
        - θ_new = θ_old - α * ∇L / (√s_new + ε)
        
        The squared gradient accumulator (s) tracks recent gradient magnitudes.
        """
        updated_params = {}
        
        for name in params:
            if name in grads:
                # Initialize squared gradient accumulator if needed
                if name not in self.squared_grads:
                    self.squared_grads[name] = np.zeros_like(params[name])
                
                # Update running average of squared gradients
                self.squared_grads[name] = (
                    self.decay_rate * self.squared_grads[name] +
                    (1 - self.decay_rate) * grads[name] ** 2
                )
                
                # Compute adaptive learning rate
                # Divide by root mean square of recent gradients
                adaptive_lr = (
                    self.learning_rate / 
                    (np.sqrt(self.squared_grads[name]) + self.epsilon)
                )
                
                # Update parameters with adaptive learning rate
                updated_params[name] = params[name] - adaptive_lr * grads[name]
            else:
                updated_params[name] = params[name]
        
        return updated_params
    
    def reset(self):
        """Reset squared gradient accumulators"""
        self.squared_grads = {}


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam)
    
    Update rule:
    - m = β₁ * m + (1-β₁) * ∇L        (first moment: mean)
    - v = β₂ * v + (1-β₂) * ∇L²       (second moment: variance)
    - m̂ = m / (1-β₁ᵗ)                 (bias correction)
    - v̂ = v / (1-β₂ᵗ)                 (bias correction)
    - θ = θ - α * m̂ / (√v̂ + ε)
    
    Mathematical Properties:
    - Combines momentum (first moment) with RMSprop (second moment)
    - Computes adaptive learning rates for each parameter
    - Includes bias correction for moment estimates
    
    When to use:
    - Default choice for most deep learning problems
    - When you want robust performance without tuning
    - Complex architectures (CNNs, Transformers, etc.)
    
    Advantages:
    - Usually works well with default hyperparameters
    - Combines benefits of momentum and RMSprop
    - Efficient and easy to implement
    - Works well in practice across many problems
    
    Why Adam is popular:
    - "Just works" for most problems
    - Less sensitive to learning rate than SGD
    - Adapts to parameter scales automatically
    - Handles sparse gradients well
    - Good convergence properties
    
    Intuition:
    - First moment (m): Where are we usually going? (momentum)
    - Second moment (v): How volatile is the journey? (RMSprop)
    - Bias correction: Fixes initialization bias in early iterations
    - Result: Adaptive, smooth, reliable optimization
    
    Typical hyperparameters:
    - learning_rate: 0.001 (good default for most problems)
    - beta1: 0.9 (momentum decay)
    - beta2: 0.999 (RMSprop decay)
    - epsilon: 1e-8 (numerical stability)
    
    Historical note:
    - Published in 2014 by Kingma & Ba
    - Became default optimizer for many frameworks
    - Name from "Adaptive Moment"
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Parameters:
        learning_rate : float - Step size (α)
        beta1 : float - Exponential decay rate for first moment (momentum)
        beta2 : float - Exponential decay rate for second moment (RMSprop)
        epsilon : float - Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (mean of gradients)
        self.v = {}  # Second moment (variance of gradients)
        self.t = 0   # Time step for bias correction
    
    def update(self, params, grads):
        """
        Update parameters using Adam optimizer.
        
        Formula:
        1. Update biased first moment:  m = β₁*m + (1-β₁)*∇L
        2. Update biased second moment: v = β₂*v + (1-β₂)*∇L²
        3. Bias correction: m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)
        4. Update parameters: θ = θ - α*m̂/(√v̂ + ε)
        """
        self.t += 1  # Increment time step
        updated_params = {}
        
        for name in params:
            if name in grads:
                # Initialize moments if needed
                if name not in self.m:
                    self.m[name] = np.zeros_like(params[name])
                    self.v[name] = np.zeros_like(params[name])
                
                # Update biased first moment (momentum)
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
                
                # Update biased second moment (RMSprop)
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2
                
                # Bias correction
                # Early iterations have biased estimates (initialized at 0)
                # Correction factor: 1/(1-β^t) grows large initially, then approaches 1
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                # Update parameters with bias-corrected adaptive learning rate
                updated_params[name] = (
                    params[name] - 
                    self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                )
            else:
                updated_params[name] = params[name]
        
        return updated_params
    
    def reset(self):
        """Reset optimizer state"""
        self.m = {}
        self.v = {}
        self.t = 0


class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm
    
    Update rule:
    - G = G + ∇L²
    - θ = θ - α * ∇L / (√G + ε)
    
    Mathematical Properties:
    - Adapts learning rate based on cumulative gradient history
    - Parameters with large cumulative gradients get small learning rates
    - Parameters with small cumulative gradients get large learning rates
    
    When to use:
    - Sparse data (NLP, recommendation systems)
    - When different features have very different frequencies
    - Early deep learning (less common now, superseded by Adam)
    
    Advantages:
    - No need to manually tune learning rate
    - Works well with sparse gradients
    - Each parameter gets its own learning rate
    
    Drawbacks:
    - Accumulates squared gradients forever (never forgets)
    - Learning rate monotonically decreases (can stop learning too early)
    - Can become too conservative in later training
    
    Why it's less used now:
    - RMSprop and Adam fix the "never forgetting" issue
    - Using exponential moving average instead of sum
    - But still useful for specific sparse data problems
    
    Intuition:
    - Frequently updated parameters: reduce their learning rate
    - Infrequently updated parameters: keep learning rate high
    - Great for dealing with sparse features
    - Think: "be careful with what you've seen often, 
              explore what you've rarely seen"
    
    Typical hyperparameters:
    - learning_rate: 0.01 (can be higher than Adam)
    - epsilon: 1e-8 (numerical stability)
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """
        Parameters:
        learning_rate : float - Base learning rate
        epsilon : float - Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated_grads = {}
    
    def update(self, params, grads):
        """
        Update parameters using AdaGrad.
        
        Formula:
        - G_new = G_old + ∇L²
        - θ_new = θ_old - α * ∇L / (√G_new + ε)
        
        Unlike RMSprop, this SUMS all squared gradients (no decay).
        """
        updated_params = {}
        
        for name in params:
            if name in grads:
                # Initialize accumulated gradients if needed
                if name not in self.accumulated_grads:
                    self.accumulated_grads[name] = np.zeros_like(params[name])
                
                # Accumulate squared gradients (no decay, keeps forever)
                self.accumulated_grads[name] += grads[name] ** 2
                
                # Compute adaptive learning rate
                # Note: As G grows, learning rate shrinks
                adaptive_lr = (
                    self.learning_rate / 
                    (np.sqrt(self.accumulated_grads[name]) + self.epsilon)
                )
                
                # Update parameters with adaptive learning rate
                updated_params[name] = params[name] - adaptive_lr * grads[name]
            else:
                updated_params[name] = params[name]
        
        return updated_params
    
    def reset(self):
        """Reset accumulated gradients"""
        self.accumulated_grads = {}


# Convenience dictionary for easy optimizer lookup
OPTIMIZERS = {
    'sgd': SGD,
    'momentum': SGDMomentum,
    'sgd_momentum': SGDMomentum,
    'rmsprop': RMSprop,
    'adam': Adam,
    'adagrad': AdaGrad
}


def get_optimizer(name, **kwargs):
    """
    Get optimizer by name with optional hyperparameters.
    
    Parameters:
    name : str or Optimizer - Name of optimizer or optimizer instance
    **kwargs : dict - Optimizer hyperparameters (learning_rate, etc.)
    
    Returns:
    Optimizer instance
    
    Example:
    >>> opt = get_optimizer('adam', learning_rate=0.001)
    >>> opt = get_optimizer('sgd', learning_rate=0.01)
    >>> opt = get_optimizer(Adam(learning_rate=0.001))
    """
    if isinstance(name, Optimizer):
        return name
    
    if isinstance(name, str):
        name = name.lower()
        if name in OPTIMIZERS:
            return OPTIMIZERS[name](**kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {name}. Choose from {list(OPTIMIZERS.keys())}")
    
    raise TypeError(f"Optimizer must be string or Optimizer instance, got {type(name)}")
