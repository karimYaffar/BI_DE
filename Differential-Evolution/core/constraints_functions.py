from typing import Callable, List
import numpy as np

class ConstriantsFunctionsHandler:
    
    @staticmethod
    def sum_of_violations(g_functions: List[Callable] = [], h_functions: List[Callable] = [], individual:np.ndarray = np.ndarray([])) -> float:
        
        sum_inequality = sum(np.maximum(0, g(individual)) ** 2 for g in g_functions)

        sum_equality = sum(abs(h(individual)) for h in h_functions)
        
        return sum_inequality + sum_equality
        
    @staticmethod
    def a_is_better_than_b_deb(a_fitness: float, a_violations: float, b_fitness: float, b_violations: float) -> bool:
        if a_violations == 0 and b_violations == 0:
            if a_fitness <= b_fitness:
                return True
            else:
                return False
        # Regla 3: Entre dos soluciones infactibles se elige el menor valor en la funcion objectivo
        elif a_violations != 0 and b_violations != 0:
            if a_violations <= b_violations:
                return True
            else:
                return False
        elif a_violations == 0 and b_violations != 0:
            return  True
        else:
            return False