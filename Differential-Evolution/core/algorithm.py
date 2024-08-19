from typing import Callable, Tuple, List
from utils.constants import SIZE_POPULATION
import numpy as np

class Algorithm:
    
    def _generate_individual_(self, upper: List[float], lower: List[float]) -> np.ndarray:
        return np.random.uniform(upper, lower)
    
    def generate(self, upper: List[float], lower: List[float]) -> np.ndarray:
        population = np.zeros((SIZE_POPULATION, len(lower)))
        
        for i in range(SIZE_POPULATION):
                population[i] = self._generate_individual_(upper, lower)

        return population

    def isValid(self, upper: List[float], lower: List[float], indiviual: np.ndarray) -> bool:
        for u, i, l in zip(upper, indiviual, lower):
            if not (l <= i <= u):
                return False
        return True