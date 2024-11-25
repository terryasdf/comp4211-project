from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        """init"""
        pass

    @abstractmethod
    def load_and_fit_data(self, data, split_size, total_size=None):
        """Load and fit data to model. Set `total_size` to None for full set"""
        pass

    @abstractmethod
    def train(self):
        """train"""
        pass

    @abstractmethod
    def evaluate(self):
        """evaluate"""
        pass