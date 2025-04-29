import numpy as np
from collections import Counter
from models.model_interface import ModelInterface

# euclidean distance function
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

class KNNModel(ModelInterface):
    def __init__(self, k):
        self.k = k
        self.data = []

    def fit(self, X, y): #knn doesn't "train" a model so we dont do anything here
        self.data = list(zip(X, y))

    def predict(self, X_queries):
        predictions = []
        for query in X_queries: # for each feature, find distance between each point and then get the k closest
            distances = [(euclidean_distance(point, query), label[0]) for point, label in self.data]
            distances.sort()
            k_nearest_labels = [label for _, label in distances[:self.k]]
            highest = Counter(k_nearest_labels).most_common(1)[0][0] # take the value from the k nearest points
            predictions.append(highest)
        return predictions
        