# examples/classification_examples.py
"""
Classification Examples

Demonstrates logistic regression, decision trees, KNN, SVM, and Naive Bayes
for binary and multi-class classification problems.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from axiom.linear_model import LogisticRegression
from axiom.tree import DecisionTreeClassifier
from axiom.neighbors import KNeighborsClassifier
from axiom.svm import SVC
from axiom.naive_bayes import GaussianNB


def generate_binary_classification_data(n_samples=200, random_state=42):
    """Generate simple 2D binary classification dataset"""
    np.random.seed(random_state)
    
    # Class 0: centered at (-2, -2)
    X0 = np.random.randn(n_samples//2, 2) + np.array([-2, -2])
    y0 = np.zeros(n_samples//2)
    
    # Class 1: centered at (2, 2)
    X1 = np.random.randn(n_samples//2, 2) + np.array([2, 2])
    y1 = np.ones(n_samples//2)
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def generate_multiclass_data(n_samples=300, n_classes=3, random_state=42):
    """Generate multi-class classification dataset"""
    np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []
    
    # Generate data for each class in different regions
    for i in range(n_classes):
        # Place each class in a different quadrant
        angle = 2 * np.pi * i / n_classes
        center = 3 * np.array([np.cos(angle), np.sin(angle)])
        
        X_class = np.random.randn(samples_per_class, 2) + center
        y_class = np.full(samples_per_class, i)
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def example_1_logistic_regression():
    """
    Example 1: Logistic Regression
    Classic algorithm for binary classification
    """
    print("=" * 60)
    print("Example 1: Logistic Regression")
    print("=" * 60)
    
    # Generate data
    X, y = generate_binary_classification_data(n_samples=200)
    
    # Split
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    model = LogisticRegression(learning_rate=0.1, max_iters=1000)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print(f"Dataset: {len(X)} samples, 2 features, 2 classes")
    print(f"Algorithm: Logistic Regression")
    print()
    print(f"Model Details:")
    print(f"  Learning rate: {model.lr}")
    print(f"  Iterations: {len(model.loss_history)}")
    print(f"  Final loss: {model.loss_history[-1]:.4f}")
    print()
    print(f"Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()
    print("Key Points:")
    print("  • Uses sigmoid function to output probabilities")
    print("  • Optimized with gradient descent")
    print("  • Good for linearly separable data")
    print("  • Fast training and prediction")
    print()


def example_2_decision_tree():
    """
    Example 2: Decision Tree Classifier
    Tree-based model with interpretable rules
    """
    print("=" * 60)
    print("Example 2: Decision Tree Classifier")
    print("=" * 60)
    
    # Generate data
    X, y = generate_binary_classification_data(n_samples=200)
    
    # Split
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train with different depths
    depths = [3, 5, 10, None]
    
    print(f"Dataset: {len(X)} samples, 2 features, 2 classes")
    print(f"Algorithm: Decision Tree")
    print()
    print("Comparing different tree depths:")
    print()
    
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, criterion='gini')
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        depth_str = str(depth) if depth else "unlimited"
        print(f"Max Depth = {depth_str:10s}:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        
        if train_acc > 0.95 and test_acc < 0.85:
            print(f"  → Overfitting (train >> test)")
        elif train_acc < 0.85 and test_acc < 0.85:
            print(f"  → Underfitting (both low)")
        else:
            print(f"  → Good balance")
        print()
    
    print("Key Points:")
    print("  • Creates interpretable decision rules")
    print("  • Can overfit with deep trees (use max_depth)")
    print("  • Handles non-linear relationships naturally")
    print("  • No feature scaling needed")
    print()


def example_3_knn():
    """
    Example 3: K-Nearest Neighbors
    Instance-based learning, no training phase
    """
    print("=" * 60)
    print("Example 3: K-Nearest Neighbors (KNN)")
    print("=" * 60)
    
    # Generate data
    X, y = generate_binary_classification_data(n_samples=200)
    
    # Split
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Test different k values
    k_values = [1, 3, 5, 10, 20]
    
    print(f"Dataset: {len(X)} samples, 2 features, 2 classes")
    print(f"Algorithm: K-Nearest Neighbors")
    print()
    print("Comparing different k values:")
    print()
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        print(f"k = {k:2d}: Accuracy = {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        if k == 1:
            print(f"      → May overfit (sensitive to noise)")
        elif k >= 10:
            print(f"      → May underfit (too many neighbors)")
        else:
            print(f"      → Good choice")
    
    print()
    print("Key Points:")
    print("  • No training phase (lazy learner)")
    print("  • Predictions can be slow for large datasets")
    print("  • Sensitive to feature scaling")
    print("  • k controls bias-variance tradeoff")
    print()


def example_4_svm():
    """
    Example 4: Support Vector Machine
    Maximum margin classification
    """
    print("=" * 60)
    print("Example 4: Support Vector Machine (SVM)")
    print("=" * 60)
    
    # Generate data
    X, y = generate_binary_classification_data(n_samples=200)
    
    # Convert labels to {-1, +1} for SVM
    y_svm = np.where(y == 0, -1, 1)
    
    # Split
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_svm[:split], y_svm[split:]
    
    # Train
    model = SVC(C=1.0, kernel='linear', max_iters=1000)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"Dataset: {len(X)} samples, 2 features, 2 classes")
    print(f"Algorithm: Support Vector Machine")
    print()
    print(f"Model Details:")
    print(f"  Kernel: Linear")
    print(f"  C parameter: {model.C} (regularization)")
    print(f"  Support vectors: {np.sum(np.abs(model.alphas) > 1e-5)} samples")
    print()
    print(f"Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()
    print("Key Points:")
    print("  • Finds maximum margin decision boundary")
    print("  • Uses support vectors (data points near boundary)")
    print("  • Kernel trick allows non-linear boundaries")
    print("  • C parameter controls regularization")
    print()


def example_5_naive_bayes():
    """
    Example 5: Naive Bayes
    Probabilistic classifier based on Bayes' theorem
    """
    print("=" * 60)
    print("Example 5: Gaussian Naive Bayes")
    print("=" * 60)
    
    # Generate data
    X, y = generate_binary_classification_data(n_samples=200)
    
    # Split
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print(f"Dataset: {len(X)} samples, 2 features, 2 classes")
    print(f"Algorithm: Gaussian Naive Bayes")
    print()
    print(f"Model Details:")
    print(f"  Classes: {model.classes_}")
    print(f"  Assumes features are independent (naive assumption)")
    print(f"  Models each feature with Gaussian distribution")
    print()
    print(f"Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()
    print("Key Points:")
    print("  • Very fast training and prediction")
    print("  • Works well with small datasets")
    print("  • Assumes feature independence (rarely true)")
    print("  • Good baseline classifier")
    print()


def example_6_multiclass_comparison():
    """
    Example 6: Multi-class Classification Comparison
    """
    print("=" * 60)
    print("Example 6: Multi-class Classification")
    print("=" * 60)
    
    # Generate 3-class data
    X, y = generate_multiclass_data(n_samples=300, n_classes=3)
    
    # Split
    split = 240
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset: {len(X)} samples, 2 features, 3 classes")
    print()
    print("Comparing classifiers on multi-class problem:")
    print()
    
    # Test different models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        print(f"{name:20s}: Accuracy = {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    print()
    print("Observations:")
    print("  • All models handle multi-class naturally")
    print("  • Decision trees work well for non-linear boundaries")
    print("  • KNN is simple but effective")
    print("  • Naive Bayes is fast but may underperform")
    print()


def example_7_model_selection_guide():
    """
    Example 7: When to use which classifier
    """
    print("=" * 60)
    print("Example 7: Classifier Selection Guide")
    print("=" * 60)
    
    print("LOGISTIC REGRESSION")
    print("  When to use:")
    print("    • Linearly separable data")
    print("    • Need probability estimates")
    print("    • Want interpretable coefficients")
    print("    • Large datasets (scales well)")
    print()
    
    print("DECISION TREE")
    print("  When to use:")
    print("    • Need interpretable rules")
    print("    • Non-linear relationships")
    print("    • Mixed feature types (numerical + categorical)")
    print("    • Don't want feature scaling")
    print()
    
    print("K-NEAREST NEIGHBORS")
    print("  When to use:")
    print("    • Small to medium datasets")
    print("    • No assumptions about data distribution")
    print("    • Need simple baseline")
    print("  Avoid when:")
    print("    • Large datasets (slow predictions)")
    print("    • Many irrelevant features")
    print()
    
    print("SUPPORT VECTOR MACHINE")
    print("  When to use:")
    print("    • Clear margin of separation")
    print("    • High-dimensional data")
    print("    • Kernel trick for non-linear boundaries")
    print("  Avoid when:")
    print("    • Very large datasets (training is slow)")
    print()
    
    print("NAIVE BAYES")
    print("  When to use:")
    print("    • Small datasets")
    print("    • Text classification")
    print("    • Need very fast training/prediction")
    print("    • Features are (approximately) independent")
    print()
    
    print("GENERAL ADVICE:")
    print("  1. Start with Logistic Regression (good baseline)")
    print("  2. Try Decision Tree if non-linear")
    print("  3. Use KNN for small datasets")
    print("  4. Try SVM for complex boundaries")
    print("  5. Use ensemble methods (Random Forest) for best performance")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CLASSIFICATION EXAMPLES")
    print("=" * 60 + "\n")
    
    example_1_logistic_regression()
    example_2_decision_tree()
    example_3_knn()
    example_4_svm()
    example_5_naive_bayes()
    example_6_multiclass_comparison()
    example_7_model_selection_guide()
    
    print("=" * 60)
    print("All classification examples completed!")
    print("=" * 60)
