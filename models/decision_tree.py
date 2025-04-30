from models.model_interface import ModelInterface
import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeModel(ModelInterface):
    def __init__(self):
        self.root = None

    def fit(self, X, y): # Build the tree
        # First ensure that our y input does not contain any nested lists
        temp = []
        for target in y:
            if isinstance(target, list): # if the index is a list, add the first item
                temp.append(target[0])  
            else:
                temp.append(target)
        y = temp
        self.root = build_tree(X, y)

    def predict(self, X): # Predict what a value will be on the tree for each x 
        return [traverse_tree(self.root, x) for x in X]

# Calculates entropy of given label
# Although we used entropy for a test, we will not any more since entropy is mainly used for 
# classification, and we are doing regression
def entropy(y):
    # Get the amount each label appears
    counts = {}
    for label in y:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    # Calculate the entropy for each value in counts
    total = len(y)
    entropy = 0.0
    for count in counts.values():
        entropy -= (count / total) * np.log2(count / total)
    return entropy

# Function to compute information gain. Information gain finds out how much entropy 
# is lost after splitting a specific node
# This is the old information_gain method that we used with our entropy model
def information_gain(X, y, feature_index, threshold):
    # Split the data based on the threshold
    left_y = [y[i] for i in range(len(X)) if X[i][feature_index] <= threshold]
    right_y = [y[i] for i in range(len(X)) if X[i][feature_index] > threshold]
    
    # Get all entropies. We subtract parent entropy from left and right
    parent_entropy = entropy(y)
    left_entropy = entropy(left_y)
    right_entropy = entropy(right_y)
    
    # Find and return the information gain. Information gain formula is
    # the entropy of the original dataset minus the sum of the weighted entropies of each subset after the split
    ig = parent_entropy - ((len(left_y) / len(y)) * left_entropy + (len(right_y) / len(y)) * right_entropy)
    return ig

# Using variance reduction now instead of entropy
def variance(y):
    return np.var(y) if len(y) > 0 else 0

# This follows the exact same format as information gain, we are just using variance instead
def variance_reduction(X, y, feature_index, threshold):
    left_y = [y[i] for i in range(len(X)) if X[i][feature_index] <= threshold]
    right_y = [y[i] for i in range(len(X)) if X[i][feature_index] > threshold]

    total_var = variance(y)
    left_var = variance(left_y)
    right_var = variance(right_y)

    return total_var (len(left_y) / len(y)) * left_var + (len(right_y) / len(y)) * right_var

# We find the best feature to use based on each features variance
def find_best_split(X, y):
    best_score = -float('inf')
    best_feature = None
    best_threshold = None

    # We will loop through each feature and get the variance for each one
    # if it is better than the previous one, replace and we use that feature
    features = len(X[0])
    for feature in range(features):
        # We get the value at each feature and test each as a threshold
        feature_values = [x[feature] for x in X]
        thresholds = sorted(set(feature_values)) 
        for threshold in thresholds:
            # Calling variance_reduction here instead of information_gain (entropy)
            score = information_gain(X, y, feature, threshold)
            if score > best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

# Recursively builds the decision tree
def build_tree(X, y):
    # the length of y will be 1 if we stop getting new values
    # which means building the the tree more will not improve it
    if len(set(y)) == 1: return DecisionTreeNode(value=y[0])

    # Find the best feature to split on and its threshold value
    feature_index, threshold = find_best_split(X, y)

    # If there is no feature then we return a node that is the average of all the remaining features
    if feature_index is None: return DecisionTreeNode(value=np.mean(y))

    # Split data based on the best threshold
    left_indices = [i for i in range(len(X)) if X[i][feature_index] <= threshold]
    right_indices = [i for i in range(len(X)) if X[i][feature_index] > threshold]

    # If one of the sides is empty return the average of the remaining features
    if not left_indices or not right_indices: return DecisionTreeNode(value=np.mean(y))

    X_left = [X[i] for i in left_indices]
    y_left = [y[i] for i in left_indices]
    X_right = [X[i] for i in right_indices]
    y_right = [y[i] for i in right_indices]

    left_child = build_tree(X_left, y_left)
    right_child = build_tree(X_right, y_right)

    return DecisionTreeNode(
        feature_index=feature_index,
        threshold=threshold,
        left=left_child,
        right=right_child
    )

# Traverse the tree to make a prediction for a given query
def traverse_tree(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature_index] <= node.threshold:
        return traverse_tree(node.left, x)
    else:
        return traverse_tree(node.right, x)