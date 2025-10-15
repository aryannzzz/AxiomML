# axiom/ensemble/random_forest_classifier.py
import numpy as np
from collections import Counter
from ..tree.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Random Forest classifier from first principles.
    Ensemble of decision trees using bagging and random feature selection.
    Final prediction by majority vote of all trees.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True):
        """
        Parameters:
        n_estimators : int - Number of trees in the forest
        max_depth : int - Maximum depth of each tree
        min_samples_split : int - Minimum samples to split a node
        min_samples_leaf : int - Minimum samples in a leaf
        max_features : str/int - Number of features for each split
        bootstrap : bool - Whether to use bootstrap sampling
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []  # List to store all individual trees
        self.feature_importances_ = None
        self.classes_ = None
    
    def _bootstrap_sample(self, X, y):
        """
        Create bootstrap sample (random sampling with replacement).
        This creates diversity among trees.
        """
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
        else:
            # Use all samples
            indices = np.arange(n_samples)
        
        return X[indices], y[indices]
    
    def _get_max_features(self, n_features):
        """Determine how many features to use for each split"""
        if self.max_features == 'sqrt':
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
        Build a forest of decision trees.
        Each tree is trained on different data and feature subsets.
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Determine number of features for each split
        self.max_features_ = self._get_max_features(n_features)
        
        # Initialize feature importance tracking
        feature_importance = np.zeros(n_features)
        
        # Grow multiple trees
        for i in range(self.n_estimators):
            print(f"Growing tree {i+1}/{self.n_estimators}")
            
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # Initialize decision tree with feature limitation
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features_
            )
            
            # Train the tree on bootstrap sample
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            # Update feature importance (simplified - track usage)
            # In practice, we'd need to track actual importance from each tree
            self._update_feature_importance(tree, feature_importance)
        
        # Normalize feature importances
        if np.sum(feature_importance) > 0:
            self.feature_importances_ = feature_importance / np.sum(feature_importance)
        else:
            self.feature_importances_ = feature_importance
        
        return self
    
    def _update_feature_importance(self, tree, feature_importance):
        """
        Update feature importance based on tree usage.
        This is simplified - real implementation would track actual importance scores.
        """
        # We'll do a simplified version that just counts feature usage
        # In practice, we should traverse the tree and calculate proper importance
        pass  # Placeholder for proper implementation
    
    def predict(self, X):
        """
        Predict classes by majority vote of all trees.
        Each tree 'votes' for a class, and the most popular wins.
        """
        if not self.trees:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority vote for each sample
        final_predictions = []
        for i in range(X.shape[0]):
            # Get votes from all trees for this sample
            votes = tree_predictions[:, i]
            # Find most common vote
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities based on tree votes.
        Probability = (number of trees voting for class) / (total trees)
        """
        if not self.trees:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize probability matrix
        proba = np.zeros((n_samples, n_classes))
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        for i in range(n_samples):
            # Count votes for each class
            votes = tree_predictions[:, i]
            vote_counts = Counter(votes)
            
            # Calculate probabilities
            total_trees = len(self.trees)
            for j, cls in enumerate(self.classes_):
                proba[i, j] = vote_counts.get(cls, 0) / total_trees
        
        return proba
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_tree_depths(self):
        """Get depths of all trees in the forest"""
        return [tree.get_depth() for tree in self.trees]
