# axiom/tree/decision_tree_regressor.py
import numpy as np

class DecisionTreeRegressor:
    """
    Decision Tree for Regression from first principles.
    Builds a tree by recursively splitting data to minimize variance.
    Each leaf node predicts the mean of target values in that region.
    """
    
    class Node:
        """Inner class representing a node in the decision tree"""
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index  # Which feature to split on
            self.threshold = threshold          # Threshold value for splitting
            self.left = left                    # Left child node (samples <= threshold)
            self.right = right                  # Right child node (samples > threshold)  
            self.value = value                  # Prediction value for leaf nodes
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Parameters:
        max_depth : int - Maximum depth of the tree. Prevents overfitting.
        min_samples_split : int - Minimum samples required to split a node.
        min_samples_leaf : int - Minimum samples required in a leaf node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_features_ = None
    
    def _mse(self, y):
        """
        Calculate Mean Squared Error for a set of target values.
        Used as the splitting criterion - we want to minimize MSE.
        """
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        Exhaustively checks all possible splits to find the one that minimizes MSE.
        """
        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        best_mse = float('inf')
        
        # If no improvement possible, return no split
        if n_samples <= 1:
            return best_feature, best_threshold
        
        # Current MSE before splitting (we want to improve this)
        current_mse = self._mse(y)
        
        # Try every feature and every possible threshold
        for feature_idx in range(n_features):
            # Get unique values of this feature as potential thresholds
            feature_values = np.unique(X[:, feature_idx])
            
            # Try thresholds between unique values
            thresholds = []
            for i in range(len(feature_values) - 1):
                thresholds.append((feature_values[i] + feature_values[i + 1]) / 2)
            
            for threshold in thresholds:
                # Split data based on this threshold
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                left_y, right_y = y[left_mask], y[right_mask]
                
                # Skip if split doesn't meet minimum samples requirement
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted MSE after split
                left_mse = self._mse(left_y)
                right_mse = self._mse(right_y)
                n_left, n_right = len(left_y), len(right_y)
                total_mse = (n_left * left_mse + n_right * right_mse) / n_samples
                
                # If this split gives better MSE, update best split
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # Only return a split if it actually improves MSE
        if best_mse < current_mse:
            return best_feature, best_threshold
        else:
            return None, None
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        This is where the magic happens - the tree grows by finding optimal splits.
        """
        n_samples, n_features = X.shape
        
        # Stopping conditions for recursion:
        # 1. Reached maximum depth
        # 2. Too few samples to split
        # 3. Can't find a good split that improves MSE
        # 4. All target values are the same (variance = 0)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           np.var(y) == 0:
            
            # Create a leaf node that predicts the mean of targets
            return self.Node(value=np.mean(y))
        
        # Find the best split for current node
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # If no good split found, make this a leaf node
        if best_feature is None:
            return self.Node(value=np.mean(y))
        
        # Split the data using the best split found
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Return a decision node (not a leaf)
        return self.Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X, y):
        """
        Build the decision tree from training data.
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, node):
        """
        Predict a single sample by traversing the tree from root to leaf.
        """
        # If we've reached a leaf node, return its prediction value
        if node.value is not None:
            return node.value
        
        # Otherwise, go left or right based on the feature threshold
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X):
        """
        Predict target values for multiple samples.
        """
        if self.root is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def get_depth(self, node=None):
        """Calculate the depth of the tree"""
        if node is None:
            node = self.root
        
        if node.value is not None:  # Leaf node
            return 1
        else:  # Decision node
            return 1 + max(self.get_depth(node.left), self.get_depth(node.right))
