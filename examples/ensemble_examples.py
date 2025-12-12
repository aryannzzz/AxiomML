# examples/ensemble_examples.py
"""
Ensemble Methods Examples

Demonstrates Random Forests for classification and regression.
Shows how ensemble methods improve upon single decision trees.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from axiom.ensemble import RandomForestClassifier, RandomForestRegressor
from axiom.tree import DecisionTreeClassifier, DecisionTreeRegressor


def generate_classification_data(n_samples=300, random_state=42):
    """Generate classification data with some noise"""
    np.random.seed(random_state)
    
    # Create a more complex dataset with overlap
    n_per_class = n_samples // 2
    
    # Class 0: cluster around (-2, -2)
    X0 = np.random.randn(n_per_class, 2) * 1.5 + np.array([-2, -2])
    y0 = np.zeros(n_per_class)
    
    # Class 1: cluster around (2, 2) with more spread
    X1 = np.random.randn(n_per_class, 2) * 1.5 + np.array([2, 2])
    y1 = np.ones(n_per_class)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def generate_regression_data(n_samples=200, random_state=42):
    """Generate regression data with non-linear pattern"""
    np.random.seed(random_state)
    
    X = np.random.uniform(-5, 5, (n_samples, 2))
    
    # Non-linear function: y = sin(x1) + 0.5*x2 + noise
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.normal(0, 0.3, n_samples)
    
    return X, y


def example_1_single_tree_vs_forest_classification():
    """
    Example 1: Single Decision Tree vs Random Forest (Classification)
    """
    print("=" * 60)
    print("Example 1: Decision Tree vs Random Forest (Classification)")
    print("=" * 60)
    
    # Generate data
    X, y = generate_classification_data(n_samples=300)
    
    # Split
    split = 240
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset: {len(X)} samples, 2 features, 2 classes")
    print()
    
    # Single Decision Tree
    print("1. Single Decision Tree:")
    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(X_train, y_train)
    
    tree_train_acc = tree.score(X_train, y_train)
    tree_test_acc = tree.score(X_test, y_test)
    
    print(f"   Train Accuracy: {tree_train_acc:.4f} ({tree_train_acc*100:.1f}%)")
    print(f"   Test Accuracy:  {tree_test_acc:.4f} ({tree_test_acc*100:.1f}%)")
    print()
    
    # Random Forest
    print("2. Random Forest (50 trees):")
    forest = RandomForestClassifier(n_estimators=50, max_depth=10)
    forest.fit(X_train, y_train)
    
    forest_train_acc = forest.score(X_train, y_train)
    forest_test_acc = forest.score(X_test, y_test)
    
    print(f"   Train Accuracy: {forest_train_acc:.4f} ({forest_train_acc*100:.1f}%)")
    print(f"   Test Accuracy:  {forest_test_acc:.4f} ({forest_test_acc*100:.1f}%)")
    print()
    
    # Comparison
    improvement = (forest_test_acc - tree_test_acc) * 100
    print(f"Improvement: {improvement:+.1f} percentage points")
    print()
    
    print("Why Random Forest is Better:")
    print("  • Reduces overfitting through averaging")
    print("  • More stable predictions")
    print("  • Less sensitive to noise in training data")
    print("  • Each tree sees different subset of data (bootstrap)")
    print()


def example_2_single_tree_vs_forest_regression():
    """
    Example 2: Single Decision Tree vs Random Forest (Regression)
    """
    print("=" * 60)
    print("Example 2: Decision Tree vs Random Forest (Regression)")
    print("=" * 60)
    
    # Generate data
    X, y = generate_regression_data(n_samples=200)
    
    # Split
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset: {len(X)} samples, 2 features, continuous target")
    print()
    
    # Single Decision Tree
    print("1. Single Decision Tree:")
    tree = DecisionTreeRegressor(max_depth=10)
    tree.fit(X_train, y_train)
    
    tree_train_r2 = tree.score(X_train, y_train)
    tree_test_r2 = tree.score(X_test, y_test)
    
    print(f"   Train R²: {tree_train_r2:.4f}")
    print(f"   Test R²:  {tree_test_r2:.4f}")
    print()
    
    # Random Forest
    print("2. Random Forest (50 trees):")
    forest = RandomForestRegressor(n_estimators=50, max_depth=10)
    forest.fit(X_train, y_train)
    
    forest_train_r2 = forest.score(X_train, y_train)
    forest_test_r2 = forest.score(X_test, y_test)
    
    print(f"   Train R²: {forest_train_r2:.4f}")
    print(f"   Test R²:  {forest_test_r2:.4f}")
    print()
    
    # Comparison
    improvement = (forest_test_r2 - tree_test_r2)
    print(f"R² Improvement: {improvement:+.4f}")
    print()
    
    print("Key Insight:")
    print("  Random Forest smooths out predictions by averaging,")
    print("  leading to better generalization on new data.")
    print()


def example_3_number_of_trees():
    """
    Example 3: Effect of number of trees in Random Forest
    """
    print("=" * 60)
    print("Example 3: Effect of Number of Trees")
    print("=" * 60)
    
    # Generate data
    X, y = generate_classification_data(n_samples=300)
    
    # Split
    split = 240
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset: {len(X)} samples, classification task")
    print()
    print("Testing different numbers of trees:")
    print()
    
    n_trees_list = [1, 5, 10, 25, 50, 100]
    
    for n_trees in n_trees_list:
        forest = RandomForestClassifier(
            n_estimators=n_trees, 
            max_depth=10
        )
        forest.fit(X_train, y_train)
        
        test_acc = forest.score(X_test, y_test)
        
        print(f"{n_trees:3d} trees: Accuracy = {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    print()
    print("Observations:")
    print("  • Performance improves with more trees (up to a point)")
    print("  • Diminishing returns after 50-100 trees")
    print("  • More trees = slower training but usually better results")
    print("  • Typical choice: 50-100 trees for most problems")
    print()


def example_4_max_depth_effect():
    """
    Example 4: Effect of max_depth on Random Forest
    """
    print("=" * 60)
    print("Example 4: Effect of Tree Depth")
    print("=" * 60)
    
    # Generate data
    X, y = generate_classification_data(n_samples=300)
    
    # Split
    split = 240
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset: {len(X)} samples, Random Forest with 50 trees")
    print()
    print("Testing different max depths:")
    print()
    
    depths = [3, 5, 10, 15, None]
    
    for depth in depths:
        forest = RandomForestClassifier(
            n_estimators=50, 
            max_depth=depth
        )
        forest.fit(X_train, y_train)
        
        train_acc = forest.score(X_train, y_train)
        test_acc = forest.score(X_test, y_test)
        
        depth_str = str(depth) if depth else "unlimited"
        print(f"Max Depth = {depth_str:10s}:")
        print(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}", end="")
        
        if test_acc < 0.80:
            print(" → May be underfitting")
        elif train_acc > 0.98 and test_acc < 0.88:
            print(" → May be overfitting")
        else:
            print(" → Good balance")
    
    print()
    print("Guidelines:")
    print("  • Shallow trees (3-5): Prevent overfitting, may underfit")
    print("  • Medium trees (10-15): Usually good balance")
    print("  • Deep trees (20+): Risk overfitting, slower training")
    print()


def example_5_feature_importance():
    """
    Example 5: Feature importance in Random Forest
    """
    print("=" * 60)
    print("Example 5: Feature Importance")
    print("=" * 60)
    
    # Generate data with clear feature importance pattern
    np.random.seed(42)
    n_samples = 300
    
    # Feature 1: very important
    f1 = np.random.randn(n_samples)
    # Feature 2: moderately important
    f2 = np.random.randn(n_samples)
    # Feature 3: not important (noise)
    f3 = np.random.randn(n_samples)
    
    X = np.column_stack([f1, f2, f3])
    
    # Target depends mainly on f1, somewhat on f2, not at all on f3
    y = (2*f1 + f2 + np.random.randn(n_samples)*0.5 > 0).astype(int)
    
    # Split
    split = 240
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train Random Forest
    forest = RandomForestClassifier(n_estimators=50, max_depth=10)
    forest.fit(X_train, y_train)
    
    print(f"Dataset: {len(X)} samples, 3 features")
    print(f"True relationship: y depends on feature 1 (strong), feature 2 (weak)")
    print(f"Feature 3 is random noise")
    print()
    
    # Note: Feature importance would need to be implemented in RandomForest
    print("Feature Importance (conceptual):")
    print("  Feature 1: High importance (main predictor)")
    print("  Feature 2: Medium importance (secondary predictor)")
    print("  Feature 3: Low importance (noise)")
    print()
    print("Use Cases:")
    print("  • Identify most predictive features")
    print("  • Guide feature engineering")
    print("  • Reduce dimensionality (drop unimportant features)")
    print("  • Gain insights into the problem")
    print()


def example_6_when_to_use_random_forests():
    """
    Example 6: When to use Random Forests
    """
    print("=" * 60)
    print("Example 6: When to Use Random Forests")
    print("=" * 60)
    
    print("ADVANTAGES:")
    print("  ✓ Excellent performance out-of-the-box")
    print("  ✓ Handles non-linear relationships")
    print("  ✓ Robust to outliers and noise")
    print("  ✓ No feature scaling needed")
    print("  ✓ Provides feature importance")
    print("  ✓ Works for classification and regression")
    print("  ✓ Less prone to overfitting than single trees")
    print()
    
    print("DISADVANTAGES:")
    print("  ✗ Slower training than single trees")
    print("  ✗ Slower predictions (need to query multiple trees)")
    print("  ✗ Less interpretable than single tree")
    print("  ✗ Larger model size (stores multiple trees)")
    print()
    
    print("BEST USE CASES:")
    print("  • Default choice for tabular data")
    print("  • Kaggle competitions")
    print("  • When you need strong performance without tuning")
    print("  • Structured/tabular data with mixed feature types")
    print("  • Medium to large datasets")
    print()
    
    print("AVOID WHEN:")
    print("  • Need very fast real-time predictions")
    print("  • Model interpretability is critical")
    print("  • Very high-dimensional sparse data (text, images)")
    print("  • Extremely large datasets (consider XGBoost)")
    print()
    
    print("HYPERPARAMETERS TO TUNE:")
    print("  1. n_estimators: Number of trees (default: 100)")
    print("     → More is better but slower")
    print()
    print("  2. max_depth: Maximum tree depth (default: None)")
    print("     → Limit to prevent overfitting")
    print()
    print("  3. min_samples_split: Min samples to split (default: 2)")
    print("     → Higher values prevent overfitting")
    print()
    print("  4. max_features: Features to consider per split")
    print("     → Adds randomness, helps decorrelate trees")
    print()


def example_7_practical_example():
    """
    Example 7: Complete practical example
    """
    print("=" * 60)
    print("Example 7: Complete Practical Workflow")
    print("=" * 60)
    
    print("TYPICAL RANDOM FOREST WORKFLOW:")
    print()
    
    print("1. PREPARE DATA")
    print("   X, y = load_data()")
    print("   X_train, X_test, y_train, y_test = train_test_split(X, y)")
    print()
    
    print("2. TRAIN BASELINE MODEL")
    print("   rf = RandomForestClassifier(n_estimators=50)")
    print("   rf.fit(X_train, y_train)")
    print()
    
    print("3. EVALUATE")
    print("   train_score = rf.score(X_train, y_train)")
    print("   test_score = rf.score(X_test, y_test)")
    print()
    
    print("4. TUNE HYPERPARAMETERS (if needed)")
    print("   • Try different n_estimators: 50, 100, 200")
    print("   • Adjust max_depth if overfitting")
    print("   • Increase min_samples_split if overfitting")
    print()
    
    print("5. MAKE PREDICTIONS")
    print("   predictions = rf.predict(X_new)")
    print()
    
    print("COMPARISON WITH OTHER METHODS:")
    print()
    print("  Single Tree → Random Forest: +3-8% accuracy")
    print("  Logistic Reg → Random Forest: +2-10% (if non-linear)")
    print("  KNN → Random Forest: +1-5% (generally more stable)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ENSEMBLE METHODS EXAMPLES - RANDOM FORESTS")
    print("=" * 60 + "\n")
    
    example_1_single_tree_vs_forest_classification()
    example_2_single_tree_vs_forest_regression()
    example_3_number_of_trees()
    example_4_max_depth_effect()
    example_5_feature_importance()
    example_6_when_to_use_random_forests()
    example_7_practical_example()
    
    print("=" * 60)
    print("All ensemble examples completed!")
    print("=" * 60)
