from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def fit(self, X, y): # The fit function should be the function that trains the model
        pass

    @abstractmethod
    def predict(self, X): # The predict function should be the function that runs the model and predicts values
        pass