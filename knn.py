# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:55:52 2025

KNN Algorithm

@author: jacob
"""

import numpy as np
from collections import Counter

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# k nearest neighbor algorithm
def knn(data, queries, k):
    # get euclidean distance of each point
    distances = [(euclidean_distance(point, queries), label) for point, label in data]
    distances.sort() # sort based on distance so we can get k labels 
    
    # get the nearest k labels
    k_nearest_labels = [label for _, label in distances[:k]]
    
    # get most common label
    label = Counter(k_nearest_labels).most_common(1)[0][0]
    
    # print out the nearest neighbors
    k_nearest = distances[:k]
    print(f"For query instance {queries} the nearest points are: {k_nearest}")
    
    return label

# weighted k nearest neighbor algorithm
def weighted_knn(data, queries, k):
    # get euclidean distance of each point
    distances = [(euclidean_distance(point, queries), label) for point, label in data]
    distances.sort() # sort based on distance so we can get k labels 
    
    # get the k nearest labels and also distance
    k_nearest = distances[:k]
    
    # print the k-nearest training points
    print(f"Query {queries} nearest neighbors' points: {k_nearest}")
    
    # calculate weighted votes for each label
    label_weights = {'+': 0, '-': 0}
    for distance, label in k_nearest:
        if distance > 0:
            weight = 1 / distance
        else:
            weight = float('inf') # if distance is 0 
        
        label_weights[label] += weight
    
    # print cumulative weighted votes for each label
    print(f"Weighted votes: + is {label_weights['+']}, - is {label_weights['-']}")
    
    # get most common label
    label = Counter(label_weights).most_common(1)[0][0]    

    return label

data = [
    ((1, 1), '+'), ((2, 4), '+'), ((3, 3), '+'), ((2, 3), '-'), ((2.5, 1), '-'), ((4, 2), '-'), ((5, 2), '-')
]
queries = [(1.5, 3), (3, 4), (4, 1)]
k = 5

results = {q: knn(data, q, k) for q in queries}
for query, label in results.items():
    print(f"Query {query} classifys as {label}")

print("\nWeighted:")

k = 5
results = {q: weighted_knn(data, q, k) for q in queries}
for query, label in results.items():
    print(f"Query {query} classifys as {label}")
