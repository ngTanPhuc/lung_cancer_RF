from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def hello(self):
        pass