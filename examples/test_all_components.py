#!/usr/bin/env python3
"""
Comprehensive test script for all AxiomML components.
Tests basic functionality and integration.
"""

import numpy as np
import sys

def test_linear_models():
    """Test all linear model implementations"""
    print("\n=== Testing Linear Models ===")
    from axiom.linear_model import (
        SimpleLinearRegression, 
        MultipleLinearRegression,
        PolynomialRegression,
        LogisticRegression
    )
    
    # Simple Linear Regression
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 4, 6, 8, 10])
    model = SimpleLinearRegression()
    model.fit(X, y)
    assert abs(model.slope - 2.0) < 0.01, "SimpleLinearRegression slope incorrect"
    assert model.score(X, y) > 0.99, "SimpleLinearRegression score too low"
    print("✓ SimpleLinearRegression working")
    
    # Multiple Linear Regression
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([5, 8, 11, 14])
    model = MultipleLinearRegression()
    model.fit(X, y)
    assert model.score(X, y) > 0.99, "MultipleLinearRegression score too low"
    assert hasattr(model, 'intercept'), "MultipleLinearRegression missing intercept property"
    assert hasattr(model, 'coef_'), "MultipleLinearRegression missing coef_ attribute"
    print("✓ MultipleLinearRegression working")
    
    # Polynomial Regression
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([1, 4, 9, 16, 25])  # y = x^2
    model = PolynomialRegression(degree=2)
    model.fit(X, y)
    assert model.score(X, y) > 0.99, "PolynomialRegression score too low"
    print("✓ PolynomialRegression working")
    
    # Logistic Regression
    X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.1, max_iters=1000)
    model.fit(X, y)
    assert model.score(X, y) >= 0.8, "LogisticRegression score too low"
    print("✓ LogisticRegression working")


def test_tree_models():
    """Test decision tree implementations"""
    print("\n=== Testing Tree Models ===")
    from axiom.tree import DecisionTreeClassifier, DecisionTreeRegressor
    
    # Decision Tree Classifier
    X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    assert model.score(X, y) >= 0.8, "DecisionTreeClassifier score too low"
    print("✓ DecisionTreeClassifier working")
    
    # Decision Tree Regressor
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)
    assert hasattr(model, 'score'), "DecisionTreeRegressor missing score method"
    assert model.score(X, y) >= 0.8, "DecisionTreeRegressor score too low"
    print("✓ DecisionTreeRegressor working")


def test_ensemble_models():
    """Test ensemble implementations"""
    print("\n=== Testing Ensemble Models ===")
    from axiom.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # Random Forest Classifier
    X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = RandomForestClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y)
    assert model.score(X, y) >= 0.8, "RandomForestClassifier score too low"
    print("✓ RandomForestClassifier working")
    
    # Random Forest Regressor
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = RandomForestRegressor(n_estimators=10, max_depth=3)
    model.fit(X, y)
    assert model.score(X, y) >= 0.8, "RandomForestRegressor score too low"
    print("✓ RandomForestRegressor working")


def test_svm():
    """Test SVM implementations"""
    print("\n=== Testing SVM ===")
    from axiom.svm import SVC, SVR
    
    # SVC
    X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 7]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    model = SVC(C=1.0, max_iters=500)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y), "SVC predictions wrong length"
    print("✓ SVC working")
    
    # SVR
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    model = SVR(C=1.0, epsilon=0.1, max_iters=500)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y), "SVR predictions wrong length"
    print("✓ SVR working")


def test_other_classifiers():
    """Test KNN and Naive Bayes"""
    print("\n=== Testing Other Classifiers ===")
    from axiom.neighbors import KNeighborsClassifier
    from axiom.naive_bayes import GaussianNB
    
    X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # KNN
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    assert model.score(X, y) >= 0.8, "KNN score too low"
    print("✓ KNeighborsClassifier working")
    
    # Gaussian NB
    model = GaussianNB()
    model.fit(X, y)
    assert model.score(X, y) >= 0.8, "GaussianNB score too low"
    print("✓ GaussianNB working")


def test_neural_networks():
    """Test neural network components"""
    print("\n=== Testing Neural Network Components ===")
    from axiom.neural_networks import (
        Dense, Dropout, BatchNorm,
        ReLU, Sigmoid, Tanh,
        SGD, Adam,
        MSELoss, BinaryCrossEntropy
    )
    
    # Test Dense layer
    layer = Dense(10, 5)
    layer.initialize('he')
    X = np.random.randn(32, 10)
    output = layer(X, training=True)
    assert output.shape == (32, 5), "Dense layer output shape incorrect"
    print("✓ Dense layer working")
    
    # Test activations
    relu = ReLU()
    X = np.array([-1, 0, 1])
    output = relu(X)
    assert np.array_equal(output, [0, 0, 1]), "ReLU not working"
    print("✓ Activation functions working")
    
    # Test optimizers
    optimizer = Adam(learning_rate=0.001)
    assert optimizer.learning_rate == 0.001, "Optimizer learning rate not set"
    print("✓ Optimizers working")
    
    # Test losses
    loss_fn = MSELoss()
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    loss = loss_fn(y_true, y_pred)
    assert loss > 0, "Loss should be positive"
    print("✓ Loss functions working")


def test_metrics():
    """Test metric functions"""
    print("\n=== Testing Metrics ===")
    from axiom.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, r2_score
    )
    
    # Classification metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    acc = accuracy_score(y_true, y_pred)
    assert 0 <= acc <= 1, "Accuracy out of range"
    
    prec = precision_score(y_true, y_pred)
    assert 0 <= prec <= 1, "Precision out of range"
    
    rec = recall_score(y_true, y_pred)
    assert 0 <= rec <= 1, "Recall out of range"
    
    f1 = f1_score(y_true, y_pred)
    assert 0 <= f1 <= 1, "F1 score out of range"
    
    print("✓ Classification metrics working")
    
    # Regression metrics
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    mse = mean_squared_error(y_true, y_pred)
    assert mse >= 0, "MSE should be non-negative"
    
    r2 = r2_score(y_true, y_pred)
    assert r2 <= 1, "R² should be at most 1"
    
    print("✓ Regression metrics working")


def test_preprocessing():
    """Test preprocessing utilities"""
    print("\n=== Testing Preprocessing ===")
    from axiom.preprocessing import PolynomialFeatures
    
    X = np.array([[1, 2], [3, 4]])
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    # With bias and degree 2, should have: 1, x1, x2, x1^2, x1*x2, x2^2 = 6 features
    assert X_poly.shape[1] == 6, f"PolynomialFeatures should create 6 features, got {X_poly.shape[1]}"
    print("✓ PolynomialFeatures working")


def main():
    """Run all tests"""
    print("=" * 60)
    print("AXIOMML COMPREHENSIVE COMPONENT TEST")
    print("=" * 60)
    
    try:
        test_linear_models()
        test_tree_models()
        test_ensemble_models()
        test_svm()
        test_other_classifiers()
        test_neural_networks()
        test_metrics()
        test_preprocessing()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
