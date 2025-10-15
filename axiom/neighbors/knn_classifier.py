# axiom/neighbors/knn_classifier.py
import numpy as np
from collections import Counter

class KNeighborsClassifier:
    """
    K-Nearest Neighbors classifier from first principles.
    Instance-based learning: no explicit training, just stores data.
    Prediction based on majority vote of k closest training examples.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Parameters:
        n_neighbors : int - Number of neighbors to consider (k)
        weights : str - 'uniform' (all neighbors equal) or 
                       'distance' (closer neighbors have more weight)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None
    
    def fit(self, X, y):
        """
        'Train' the model by storing the training data.
        KNN is a lazy learner - no computation happens during fitting!
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y).flatten()
        self.classes_ = np.unique(self.y_train)
        return self
    
    def _euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two points.
        Formula: √(Σ(x1_i - x2_i)²)
        
        Why Euclidean?
        - Most common distance metric
        - Works well when features are on similar scales
        - Intuitive geometric interpretation
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _get_neighbors(self, x):
        """
        Find the k nearest neighbors to point x.
        Returns indices and distances of nearest neighbors.
        """
        distances = []
        
        # Calculate distance to every training point
        for i, x_train in enumerate(self.X_train):
            dist = self._euclidean_distance(x, x_train)
            distances.append((i, dist))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.n_neighbors]
        
        return neighbors
    
    def _weighted_vote(self, neighbor_indices, neighbor_distances):
        """
        Perform weighted voting based on neighbor distances.
        Closer neighbors have more influence on the prediction.
        """
        class_weights = {}
        
        for idx, dist in zip(neighbor_indices, neighbor_distances):
            label = self.y_train[idx]
            
            # Avoid division by zero for identical points
            weight = 1 / (dist + 1e-8) if self.weights == 'distance' else 1
            
            if label in class_weights:
                class_weights[label] += weight
            else:
                class_weights[label] = weight
        
        # Return class with highest total weight
        return max(class_weights.items(), key=lambda x: x[1])[0]
    
    def predict(self, X):
        """
        Predict class labels for test samples.
        For each test point:
        1. Find k nearest training points
        2. Take majority vote of their labels
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted first")
        
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Get k nearest neighbors (indices and distances)
            neighbors = self._get_neighbors(x)
            neighbor_indices = [idx for idx, _ in neighbors]
            neighbor_distances = [dist for _, dist in neighbors]
            
            if self.weights == 'distance':
                # Weighted voting (closer neighbors count more)
                pred = self._weighted_vote(neighbor_indices, neighbor_distances)
            else:
                # Simple majority voting
                neighbor_labels = [self.y_train[idx] for idx in neighbor_indices]
                pred = Counter(neighbor_labels).most_common(1)[0][0]
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        Returns probability for each class based on neighbor votes.
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted first")
        
        X = np.array(X)
        probabilities = []
        
        for x in X:
            neighbors = self._get_neighbors(x)
            neighbor_indices = [idx for idx, _ in neighbors]
            neighbor_labels = [self.y_train[idx] for idx in neighbor_indices]
            
            # Count occurrences of each class
            label_counts = Counter(neighbor_labels)
            
            # Calculate probabilities
            if self.weights == 'distance':
                # Weighted probabilities
                total_weight = 0
                class_weights = {}
                
                for idx, (_, dist) in enumerate(neighbors):
                    label = neighbor_labels[idx]
                    weight = 1 / (dist + 1e-8)
                    class_weights[label] = class_weights.get(label, 0) + weight
                    total_weight += weight
                
                # Normalize to get probabilities
                prob = [class_weights.get(cls, 0) / total_weight for cls in self.classes_]
            else:
                # Uniform probabilities
                prob = [label_counts.get(cls, 0) / self.n_neighbors for cls in self.classes_]
            
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
