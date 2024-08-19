import numpy as np
from .Problem import Problem, ProblemType

INF = np.inf
EPS = 1.0e-14
E  = np.e
PI = np.pi
D = 2

SUPERIOR = np.array([100] * D)
INFERIOR = np.array([-100] * D)



class CEC2022_ZakharovF(Problem):
    def __init__(self):
        super().__init__(
            ProblemType.CONSTRAINED,
            SUPERIOR, INFERIOR,
            [], []
        )

    def fitness(self, x: np.array) -> float:
        term1 = np.sum(x**2)
        term2 = np.sum(0.5 * x)
        result = term1 + term2**2 + term2**4
        return result
    
class CEC2022_RosenbrockF(Problem):
    def __init__(self):
        super().__init__(
            tipo=ProblemType.UNCONSTRAINED, 
            superior=SUPERIOR, 
            inferior=INFERIOR, 
            rest_g=[], 
            rest_h=[]
        )
        
    def fitness(self, x: np.array) -> float:
        result = 0.0
        for i in range(D - 1):
            result += 100 * (x[i]**2 - x[i + 1])**2 + (x[i + 1] - 1)**2
        return result

