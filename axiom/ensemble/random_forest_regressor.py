# axiom/ensemble/random_forest_regressor.py
import numpy as np
from ..tree.decision_tree_regressor import DecisionTreeRegressor

class RandomForestRegressor:
    """
    Random Forest for Regression from first principles.
    Ensemble method that combines multiple decision trees.
    Uses bagging (bootstrap aggregating) and random feature selection.
    Final prediction is the average of all tree predictions.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None, bootstrap=True):
        """
        Parameters:
        n_estimators : int - Number of trees in the forest
        max_depth : int - Maximum depth of each tree
        min_samples_split : int - Minimum samples to split a node
        min_samples_leaf : int - Minimum samples in a leaf node
        max_features : int/float/str - Number of features to consider for each split
                    If 'sqrt', use sqrt(n_features)
                    If 'log2', use log2(n_features)  
                    If None, use all features
        bootstrap : bool - Whether to use bootstrap sampling
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []  # Will store all the individual decision trees
        self.feature_importances_ = None
    
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample (random sampling with replacement).
        This creates diversity among trees by giving each tree slightly different data.
        """
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Sample indices with replacement (same size as original dataset)
            indices = np.random.choice(n_samples, n_samples, replace=True)
        else:
            # Use all samples (no bootstrapping)
            indices = np.arange(n_samples)
        
        return X[indices], y[indices]
    
    def _get_random_features(self, n_features):
        """
        Select a random subset of features for each tree.
        This creates diversity and decorrelates the trees.
        """
        if self.max_features is None:
            return n_features  # Use all features
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features
    
    def fit(self, X, y):
        """
        Build a forest of trees from training data.
        Each tree is trained on a different bootstrap sample and feature subset.
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Determine how many features to use for each split
        self.max_features_ = self._get_random_features(n_features)
        
        # Initialize feature importance array
        feature_importance = np.zeros(n_features)
        
        # Grow multiple trees in parallel
        for i in range(self.n_estimators):
            print(f"Growing tree {i+1}/{self.n_estimators}")
            
            # Create bootstrap sample (with replacement)
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # Initialize decision tree with parameters
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            # For random feature selection, we'll modify how the tree finds splits
            # We'll temporarily restrict the features it can use
            original_find_split = tree._find_best_split
            
            def _random_feature_find_split(X, y):
                """Wrapper that only considers a random subset of features"""
                # Randomly select features for this tree
                feature_indices = np.random.choice(
                    n_features, self.max_features_, replace=False
                )
                
                # Only use the selected features
                X_reduced = X[:, feature_indices]
                
                # Find best split using reduced feature set
                best_feature_reduced, best_threshold = original_find_split(X_reduced, y)
                
                # Map back to original feature indices
                if best_feature_reduced is not None:
                    best_feature = feature_indices[best_feature_reduced]
                    # Track feature importance (how often a feature is used for splitting)
                    feature_importance[best_feature] += 1
                    return best_feature, best_threshold
                else:
                    return None, None
            
            # Replace the split-finding method for this tree
            tree._find_best_split = _random_feature_find_split
            
            # Train the tree on bootstrap sample
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        # Calculate feature importances (normalized)
        self.feature_importances_ = feature_importance / np.sum(feature_importance)
        
        return self
    
    def predict(self, X):
        """
        Make predictions by averaging predictions from all trees.
        This aggregation reduces variance and improves generalization.
        """
        if not self.trees:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        
        # Get predictions from all trees
        # Each tree gives a prediction for each sample
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Average predictions across all trees (ensemble prediction)
        # This is the core idea of Random Forest - wisdom of the crowd
        return np.mean(tree_predictions, axis=0)
    
    def score(self, X, y):
        """R² score for the random forest"""
        y_pred = self.predict(X)
        y = np.array(y).flatten()
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def get_tree_depths(self):
        """Get depths of all trees in the forest"""
        return [tree.get_depth() for tree in self.trees]
