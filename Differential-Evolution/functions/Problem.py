import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Callable

class ProblemType(Enum):
    CONSTRAINED = 1
    UNCONSTRAINED = 2

class Problem(ABC):
    def __init__(
        self, tipo: ProblemType,
        superior: np.array,
        inferior: np.array,
        rest_g: List[Callable[[np.array], float]],
        rest_h: List[Callable[[np.array], float]]):
        
        self.tipo = tipo
        self.superior = superior
        self.inferior = inferior
        self.rest_g = rest_g
        self.rest_h = rest_h

    @abstractmethod
    def fitness(self, x: np.array) -> float:
        pass