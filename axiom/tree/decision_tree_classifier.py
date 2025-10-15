# axiom/tree/decision_tree_classifier.py
import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    """
    Decision Tree for classification from first principles.
    Builds tree by recursively splitting data to maximize information gain.
    Uses Gini impurity or entropy as splitting criteria.
    Each leaf node predicts the most common class in that region.
    """
    
    class Node:
        """Inner class representing a node in the decision tree"""
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index  # Feature to split on
            self.threshold = threshold          # Split threshold
            self.left = left                    # Left child (<= threshold)
            self.right = right                  # Right child (> threshold)
            self.value = value                  # Predicted class (leaf nodes only)
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', max_features=None):
        """
        Parameters:
        max_depth : int - Maximum tree depth
        min_samples_split : int - Minimum samples to split a node
        min_samples_leaf : int - Minimum samples in a leaf
        criterion : str - 'gini' or 'entropy' for split quality
        max_features : int - Number of features to consider for each split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.root = None
        self.n_features_ = None
        self.n_classes_ = None
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for a set of labels.
        Measures how often a randomly chosen element would be misclassified.
        
        Formula: 1 - Σ(p_i²) where p_i is probability of class i
        
        Why Gini?
        - Computationally efficient (no logarithms)
        - Tends to create balanced splits
        """
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Calculate sum of squared probabilities
        sum_squared_probs = 0
        for count in class_counts.values():
            prob = count / total_samples
            sum_squared_probs += prob ** 2
        
        return 1 - sum_squared_probs
    
    def _entropy(self, y):
        """
        Calculate entropy for a set of labels.
        Measures the uncertainty or randomness in the data.
        
        Formula: -Σ(p_i * log2(p_i))
        
        Why Entropy?
        - More sensitive to class probability changes
        - Tends to create purer splits
        """
        if len(y) == 0:
            return 0
        
        class_counts = Counter(y)
        total_samples = len(y)
        entropy = 0
        
        for count in class_counts.values():
            prob = count / total_samples
            if prob > 0:  # log(0) is undefined
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _information_gain(self, y, y_left, y_right):
        """
        Calculate information gain from a split.
        Measures how much the split reduces impurity.
        
        Formula: IG = impurity(parent) - weighted_avg(impurity(children))
        """
        if self.criterion == 'gini':
            impurity_parent = self._gini_impurity(y)
            impurity_left = self._gini_impurity(y_left)
            impurity_right = self._gini_impurity(y_right)
        else:  # entropy
            impurity_parent = self._entropy(y)
            impurity_left = self._entropy(y_left)
            impurity_right = self._entropy(y_right)
        
        # Weighted average of child impurities
        n_left, n_right, n_total = len(y_left), len(y_right), len(y)
        weighted_impurity = (n_left / n_total) * impurity_left + (n_right / n_total) * impurity_right
        
        return impurity_parent - weighted_impurity
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        Maximizes information gain across all possible splits.
        """
        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        best_gain = -1
        
        # If we're limiting features, select random subset
        if self.max_features and self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)
        
        # Try each feature and potential threshold
        for feature_idx in feature_indices:
            # Get unique feature values as potential thresholds
            feature_values = np.unique(X[:, feature_idx])
            
            # Try thresholds between unique values
            thresholds = []
            for i in range(len(feature_values) - 1):
                thresholds.append((feature_values[i] + feature_values[i + 1]) / 2)
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                left_y, right_y = y[left_mask], y[right_mask]
                
                # Skip if split doesn't meet minimum samples requirement
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                gain = self._information_gain(y, left_y, right_y)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        Each call creates a node and potentially splits it further.
        """
        n_samples, n_features = X.shape
        
        # Stopping conditions:
        # 1. Reached maximum depth
        # 2. Too few samples to split
        # 3. All samples belong to same class (pure node)
        # 4. Can't find a split that improves information gain
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            
            # Create leaf node with most common class
            most_common_class = Counter(y).most_common(1)[0][0]
            return self.Node(value=most_common_class)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no good split found, make leaf node
        if best_feature is None or best_gain <= 0:
            most_common_class = Counter(y).most_common(1)[0][0]
            return self.Node(value=most_common_class)
        
        # Split data and recursively build subtrees
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Return decision node (can be split further)
        return self.Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X, y):
        """Build the decision tree from training data"""
        X = np.array(X)
        y = np.array(y).flatten()
        
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, node):
        """Predict class for a single sample by traversing the tree"""
        # Leaf node: return the predicted class
        if node.value is not None:
            return node.value
        
        # Decision node: go left or right based on feature threshold
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        """Predict classes for multiple samples"""
        if self.root is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        For each sample, returns probability distribution over classes.
        """
        if self.root is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        
        for i, x in enumerate(X):
            # Traverse to leaf node for this sample
            node = self.root
            while node.value is None:
                if x[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            
            # Create one-hot encoded probability vector
            # This is simplified - real implementation would track class distribution at leaves
            class_idx = np.where(self.classes_ == node.value)[0][0]
            proba[i, class_idx] = 1.0
        
        return proba
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
