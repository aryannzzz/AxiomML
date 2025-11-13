# examples/regression_examples.py
"""
Regression Examples - Linear Models

Demonstrates simple linear regression, multiple linear regression,
polynomial regression, and their usage patterns.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from axiom.linear_model import (
    SimpleLinearRegression,
    MultipleLinearRegression,
    PolynomialRegression
)


def example_1_simple_linear_regression():
    """
    Example 1: Simple Linear Regression (1 feature)
    Predicting house prices based on square footage
    """
    print("=" * 60)
    print("Example 1: Simple Linear Regression")
    print("=" * 60)
    
    # Generate synthetic data: y = 50 + 100*x + noise
    np.random.seed(42)
    X = np.random.uniform(10, 50, 100).reshape(-1, 1)  # Square footage (hundreds)
    y = 50 + 100 * X.flatten() + np.random.normal(0, 200, 100)  # Price ($1000s)
    
    # Split into train/test
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = SimpleLinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    r2 = model.score(X_test, y_test)
    
    print(f"Dataset: {len(X)} houses")
    print(f"Feature: Square footage (hundreds of sq ft)")
    print(f"Target: House price ($1000s)")
    print()
    print(f"Learned Parameters:")
    print(f"  Slope (β₁): {model.slope:.2f} → Each 100 sq ft increases price by ${model.slope:.0f}k")
    print(f"  Intercept (β₀): {model.intercept:.2f} → Base price is ${model.intercept:.0f}k")
    print()
    print(f"Performance:")
    print(f"  R² Score: {r2:.4f} ({r2*100:.1f}% of variance explained)")
    print()
    print(f"Sample Predictions:")
    for i in range(3):
        print(f"  House {i+1}: {X_test[i][0]:.0f}00 sq ft → "
              f"Actual: ${y_test[i]:.0f}k, Predicted: ${y_pred[i]:.0f}k")
    print()


def example_2_multiple_linear_regression():
    """
    Example 2: Multiple Linear Regression (multiple features)
    Predicting house prices based on multiple features
    """
    print("=" * 60)
    print("Example 2: Multiple Linear Regression")
    print("=" * 60)
    
    # Generate synthetic data with 3 features
    np.random.seed(42)
    n_samples = 100
    
    # Features: square footage, bedrooms, age
    square_footage = np.random.uniform(10, 50, n_samples)
    bedrooms = np.random.randint(1, 5, n_samples)
    age = np.random.uniform(0, 30, n_samples)
    
    X = np.column_stack([square_footage, bedrooms, age])
    
    # True relationship: price = 50 + 100*sqft + 30*bedrooms - 5*age + noise
    y = (50 + 
         100 * square_footage + 
         30 * bedrooms - 
         5 * age + 
         np.random.normal(0, 50, n_samples))
    
    # Split data
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = MultipleLinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    
    print(f"Dataset: {len(X)} houses")
    print(f"Features: Square footage, Bedrooms, Age (years)")
    print(f"Target: House price ($1000s)")
    print()
    print(f"Learned Parameters:")
    print(f"  Intercept: {model.intercept:.2f}")
    print(f"  Coefficients:")
    feature_names = ['Square Footage (100s)', 'Bedrooms', 'Age (years)']
    for name, coef in zip(feature_names, model.coefficients):
        sign = "+" if coef >= 0 else ""
        print(f"    {name:25s}: {sign}{coef:.2f}")
    print()
    print(f"Performance:")
    print(f"  R² Score: {r2:.4f} ({r2*100:.1f}% of variance explained)")
    print()
    print(f"Interpretation:")
    print(f"  • Each 100 sq ft adds ${model.coefficients[0]:.0f}k to price")
    print(f"  • Each bedroom adds ${model.coefficients[1]:.0f}k to price")
    print(f"  • Each year of age reduces price by ${abs(model.coefficients[2]):.0f}k")
    print()


def example_3_polynomial_regression():
    """
    Example 3: Polynomial Regression (non-linear relationships)
    Modeling curved relationships between variables
    """
    print("=" * 60)
    print("Example 3: Polynomial Regression")
    print("=" * 60)
    
    # Generate non-linear data: y = 5 + 2x + 3x² + noise
    np.random.seed(42)
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = 5 + 2*X.flatten() + 3*X.flatten()**2 + np.random.normal(0, 1, 100)
    
    # Split data
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Compare different polynomial degrees
    degrees = [1, 2, 3, 5]
    
    print(f"Dataset: {len(X)} samples with non-linear relationship")
    print(f"True relationship: y = 5 + 2x + 3x² + noise")
    print()
    print("Comparing polynomial degrees:")
    print()
    
    for degree in degrees:
        model = PolynomialRegression(degree=degree)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        
        print(f"Degree {degree}:")
        print(f"  R² Score: {r2:.4f}")
        
        if degree == 1:
            print(f"  → Underfitting (linear model for quadratic data)")
        elif degree == 2:
            print(f"  → Good fit (matches true relationship)")
        elif degree >= 3:
            print(f"  → Risk of overfitting (too flexible)")
        print()
    
    print("Key Insight:")
    print("  Degree 2 fits best because the true relationship is quadratic.")
    print("  Higher degrees overfit to noise, lower degrees underfit.")
    print()


def example_4_regression_comparison():
    """
    Example 4: Comparing all regression methods
    """
    print("=" * 60)
    print("Example 4: Regression Methods Comparison")
    print("=" * 60)
    
    # Generate slightly non-linear data
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 + 3*X.flatten() + 0.5*X.flatten()**2 + np.random.normal(0, 2, 100)
    
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Test different models
    models = {
        'Simple Linear': SimpleLinearRegression(),
        'Multiple Linear': MultipleLinearRegression(),
        'Polynomial (deg=2)': PolynomialRegression(degree=2),
        'Polynomial (deg=3)': PolynomialRegression(degree=3)
    }
    
    print(f"Dataset: {len(X)} samples")
    print(f"True relationship: y = 2 + 3x + 0.5x² + noise")
    print()
    print("Model Performance Comparison:")
    print()
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        results.append((name, r2))
        
        print(f"{name:25s}: R² = {r2:.4f}")
    
    print()
    best_model = max(results, key=lambda x: x[1])
    print(f"Best Model: {best_model[0]} (R² = {best_model[1]:.4f})")
    print()
    print("Conclusion:")
    print("  Polynomial regression (degree 2) captures the quadratic")
    print("  relationship best. Simple linear regression underfits.")
    print()


def example_5_practical_tips():
    """
    Example 5: Practical tips for regression
    """
    print("=" * 60)
    print("Example 5: Practical Tips & Best Practices")
    print("=" * 60)
    
    print("1. FEATURE SCALING")
    print("   • Important when features have different scales")
    print("   • Normalize or standardize before training")
    print()
    
    print("2. POLYNOMIAL DEGREE SELECTION")
    print("   • Start with degree=1 (linear)")
    print("   • Increase if underfitting (low R² on training)")
    print("   • Decrease if overfitting (good train R², poor test R²)")
    print()
    
    print("3. TRAIN/TEST SPLIT")
    print("   • Always evaluate on unseen test data")
    print("   • Typical split: 70-80% train, 20-30% test")
    print()
    
    print("4. INTERPRETING R² SCORE")
    print("   • R² = 1.0: Perfect fit (rare in real data)")
    print("   • R² = 0.8-0.9: Very good fit")
    print("   • R² = 0.6-0.8: Decent fit")
    print("   • R² < 0.5: Poor fit (consider more features or complexity)")
    print()
    
    print("5. WHEN TO USE EACH MODEL")
    print("   • Simple Linear: 1 feature, linear relationship")
    print("   • Multiple Linear: Multiple features, linear relationship")
    print("   • Polynomial: Non-linear relationship, curved data")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REGRESSION EXAMPLES - LINEAR MODELS")
    print("=" * 60 + "\n")
    
    example_1_simple_linear_regression()
    example_2_multiple_linear_regression()
    example_3_polynomial_regression()
    example_4_regression_comparison()
    example_5_practical_tips()
    
    print("=" * 60)
    print("All regression examples completed!")
    print("=" * 60)
