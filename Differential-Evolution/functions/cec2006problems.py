import numpy as np
from .Problem import Problem, ProblemType

# Problem 01
class CEC2006_G01(Problem):
    
    SUPERIOR = np.array([1,1,1,1,1,1,1,1,1,100,100,100,1])
    INFERIOR = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

    def __init__(self):
        rest_g = [
            self.cec2006_g01_g1,self.cec2006_g01_g2,self.cec2006_g01_g3,
            self.cec2006_g01_g4,self.cec2006_g01_g5,self.cec2006_g01_g6,
            self.cec2006_g01_g7,self.cec2006_g01_g8,self.cec2006_g01_g9
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g,[] #RESTRICCIONES
        )
    
    def fitness(self, individuo: np.array) -> float:
        sum1 = np.sum(individuo[0:4])
        sum2 = np.sum(individuo[0:4]**2)
        sum3 = np.sum(individuo[4:13])
        f_x = 5 * sum1 - 5 * sum2 - sum3
        return f_x

    @staticmethod
    def cec2006_g01_g1(x):
        return 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10  # restriccion 1 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g2(x):
        return 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10  # restriccion 2 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g3(x):
        return 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10 # restriccion 3 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g4(x):
        return -8 * x[0] + x[9] # restriccion 4 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g5(x):
        return -8 * x[1] + x[10] # restriccion 5 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g6(x):
        return -8 * x[2] + x[11] # restriccion 6 de desigualdad <= 0
    
    @staticmethod              
    def cec2006_g01_g7(x):
        return -2 * x[3] - x[4] + x[9] #restriccion 7 de desigualdad <= 0

    @staticmethod
    def cec2006_g01_g8(x):
        return -2 * x[5] - x[6] + x[10] # restriccion 8 de desigualdad <= 0

    @staticmethod
    def cec2006_g01_g9(x):
        return -2 * x[7] - x[8] + x[11] # restriccion 9 de desigualdad <= 0
    


#Notas: x es un arrelo de tamaño t= 13 y 0 ≤ xi ≤ 1(i = 1,...,9),0 ≤ xi ≤ 100(i = 10,11,12) 

    
#********************************************************************************************************************************

#Problem 02

class CEC2006_G02(Problem):
    SUPERIOR = np.array([10] * 20)
    INFERIOR = np.array([0] * 20)

    def __init__(self):
        rest_g = [
            self.cec2006_g02_g1,self.cec2006_g02_g2,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g,[] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        sum_cos4 = np.sum(np.cos(individuo)**4)
        prod_cos2 = np.prod(np.cos(individuo)**2)
        sum_ix2 = np.sum((np.arange(1, len(individuo) + 1) * individuo**2))
        f_x = -abs((sum_cos4 - 2 * prod_cos2) / np.sqrt(sum_ix2))
        return f_x

    @staticmethod
    def cec2006_g02_g1(x):  # restriccion 1 de desigualdad <= 0
        product_x = np.prod(x)
        result = 0.75 - product_x
        return result

    @staticmethod
    def cec2006_g02_g2(x):  # restriccion 2 de desigualdad <= 0
        sum_x = np.sum(x)
        n = len(x)
        result = sum_x - 7.5 * n
        return result


#Notas: X es un arreglo de tamaño  t=20, 0<Xi<=10, (i=1,2,...,t)

#********************************************************************************************************************************

#Problem 03

class CEC2006_G03(Problem):
    SUPERIOR = np.array([1] * 10)
    INFERIOR = np.array([0] * 10)

    def __init__(self):
        rest_h = [
            self.cec2006_g03_h1
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            [],rest_h #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        n = len(individuo)
        product_x = np.prod(individuo)
        f_x = -(np.sqrt(n)**n * product_x)
        return f_x

    @staticmethod
    def cec2006_g03_h1(x): # restriccion 1 de igualdad = 0
        sum_x_squared = np.sum(x**2) 
        return sum_x_squared - 1

#Notas: donde x es de tamaño t=10, 0 ≤ xi ≤ 1(i = 1,...,n).

#********************************************************************************************************************************

#Problem 04

class CEC2006_G04(Problem):
    
    SUPERIOR = np.array([102, 45, 45, 45, 45])
    INFERIOR = np.array([78, 33, 27, 27, 27])

    def __init__(self):
        rest_g = [
            self.cec2006_g04_g2,self.cec2006_g04_g1,self.cec2006_g04_g3,
            self.cec2006_g04_g4,self.cec2006_g04_g5,self.cec2006_g04_g6
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g,[] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = 5.3578547 * x[2]**2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141
        return f_x
    
    @staticmethod
    def cec2006_g04_g1(x): # restriccion 1 de desigualdad <= 0
        return 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92

    @staticmethod
    def cec2006_g04_g2(x): # restriccion 2 de desigualdad <= 0
        return -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4]

    @staticmethod
    def cec2006_g04_g3(x): # restriccion 3 de desigualdad <= 0
        return 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2]**2 - 110

    @staticmethod
    def cec2006_g04_g4(x): # restriccion 4 de desigualdad <= 0
        return -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2]**2 + 90

    @staticmethod
    def cec2006_g04_g5(x): # restriccion 5 de desigualdad <= 0
        return 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25

    @staticmethod
    def cec2006_g04_g6(x): # restriccion 6 de desigualdad <= 0
        return -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20


#Notas: donde 78 ≤ x1 ≤ 102, 33 ≤ x2 ≤ 45 y 27 ≤ xi ≤ 45(i = 3,4,5)

#********************************************************************************************************************************

#Problem 05

class CEC2006_G05(Problem):
    
    # WITH BOUNDS:
    # 0 ≤ x1 ≤ 1200, 0 ≤ x2 ≤ 1200, −0.55 ≤ x3 ≤ 0.55 y −0.55 ≤ x4 ≤ 0.55.
        
    SUPERIOR = np.array([1200, 1200, 0.55, 0.55])
    INFERIOR = np.array([0, 0, -0.55, -0.55])

    def __init__(self):
        rest_g = [
            self.cec2006_g05_g1,self.cec2006_g05_g2,
        ]
        rest_h = [
            self.cec2006_g05_h1,self.cec2006_g05_h2,self.cec2006_g05_h3
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g,rest_h #RESTRICCIONES
        )

    def fitness(self, individuo:np.array): 
        x = individuo
        f_x = 3 * x[0] + 0.000001* x[0]**3 + 2* x[1] + (0.000002/3) * x[1]**3
        return f_x

    @staticmethod
    def cec2006_g05_g1(x): # restriccion 1 de desigualdad <= 0
        return -x[3] + x[2] - 0.55
    
    @staticmethod
    def cec2006_g05_g2(x): # restriccion 2 de desigualdad <= 0
        return -x[2] + x[3] - 0.55
    
    @staticmethod
    def cec2006_g05_h1(x): # restriccion 3 de igualdad = 0
        return 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]
    
    @staticmethod
    def cec2006_g05_h2(x): # restriccion 4 de igualdad = 0
        return 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]
    
    @staticmethod
    def cec2006_g05_h3(x): # restriccion 5 de igualdad = 0
        return 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8


#********************************************************************************************************************************

#Problem 06
class CEC2006_G06:
    
    # WITH BOUNDS:
    # 13 ≤ x1 ≤ 100 y 0 ≤ x2 ≤ 100
        
    SUPERIOR = np.array([100, 100])
    INFERIOR = np.array([13, 0])

    def __init__(self):
        rest_g = [
            self.cec2006_g06_g1,self.cec2006_g06_g2,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g, [] #RESTRICCIONES
        )
    
    def fitness(self, individuo:np.array): 
        x = individuo
        f_x = (x[0] -10)**3 + (x[1]-20)**3
        return f_x
    
    @staticmethod
    def cec2006_g06_g1(x): # restriccion 1 de desigualdad <= 0
        return -(x[0] -5)**2 - (x[1] - 5)**2 +100 
    
    @staticmethod
    def cec2006_g06_g2(x): # restriccion 2 de desigualdad <= 0
        return (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81

#********************************************************************************************************************************

#Problem 07

import numpy as np

class CEC2006_G07(Problem):
    SUPERIOR = np.array([10] * 10)
    INFERIOR = np.array([-10] * 10)

    def __init__(self):
        rest_g = [
            self.cec2006_g07_g1,self.cec2006_g07_g2,self.cec2006_g07_g3,self.cec2006_g07_g4,
            self.cec2006_g07_g5,self.cec2006_g07_g6,self.cec2006_g07_g7,self.cec2006_g07_g8,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR, self.INFERIOR, #LÍMITES
            rest_g, [] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1] + (x[2] - 10)**2 + 4*(x[3] - 5)**2 + (x[4] - 3)**2 + 2*(x[5] - 1)**2 + 5*x[6]**2 + 7*(x[7] - 11)**2 + 2*(x[8] - 10)**2 + (x[9] - 7)**2 + 45
        return f_x

    @staticmethod
    def cec2006_g07_g1(x):  # restriccion 1 de desigualdad <= 0
        result = -105 + 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7]
        return result

    @staticmethod
    def cec2006_g07_g2(x):  # restriccion 2 de desigualdad <= 0
        result = 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]
        return result

    @staticmethod
    def cec2006_g07_g3(x):  # restriccion 3 de desigualdad <= 0
        result = -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12
        return result

    @staticmethod
    def cec2006_g07_g4(x):  # restriccion 4 de desigualdad <= 0
        result = 3*(x[0] - 2)**2 + 4*(x[1] - 3)**2 + 2*x[2]**2 - 7*x[3] - 120
        return result

    @staticmethod
    def cec2006_g07_g5(x):  # restriccion 5 de desigualdad <= 0
        result = 5*x[0]**2 + 8*x[1] + (x[2] - 6)**2 - 2*x[3] - 40
        return result

    @staticmethod
    def cec2006_g07_g6(x):  # restriccion 6 de desigualdad <= 0
        result = x[0]**2 + 2*(x[1] - 2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5]
        return result

    @staticmethod
    def cec2006_g07_g7(x):  # restriccion 7 de desigualdad <= 0
        result = 0.5*(x[0] - 8)**2 + 2*(x[1] - 4)**2 + 3*x[4]**2 - x[5] - 30
        return result

    @staticmethod
    def cec2006_g07_g8(x):  # restriccion 8 de desigualdad <= 0
        result = -3*x[0] + 6*x[1] + 12*(x[8] - 8)**2 - 7*x[9]
        return result
    
    #Notas: donde −10 ≤ xi ≤ 10(i = 1,...,10)
    
#********************************************************************************************************************************

#Problem 08

class CEC2006_G08(Problem):
    SUPERIOR = np.array([10, 10])
    INFERIOR = np.array([0, 0])

    def __init__(self):
        rest_g = [
            self.cec2006_g08_g1,self.cec2006_g08_g2,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR, self.INFERIOR, #LÍMITES
            rest_g, [] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = - (np.sin(2 * np.pi * x[0])**3 * np.sin(2 * np.pi * x[1])) / (x[0]**3 * (x[0] + x[1]))
        return f_x

    @staticmethod
    def cec2006_g08_g1(x):  # restriccion 1 de desigualdad <= 0
        result = x[0]**2 - x[1] + 1
        return result

    @staticmethod
    def cec2006_g08_g2(x):  # restriccion 2 de desigualdad <= 0
        result = 1 - x[0] + (x[1] - 4)**2
        return result
    
#********************************************************************************************************************************

#Problem 9

class CEC2006_G09:
    SUPERIOR = np.array([10] * 7)
    INFERIOR = np.array([-10] * 7)

    def __init__(self):
        rest_g = [
            self.cec2006_g09_g1,self.cec2006_g09_g2,self.cec2006_g09_g3,self.cec2006_g09_g4,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR, self.INFERIOR, #LÍMITES
            rest_g, [] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = (x[0] - 10)**2 + 5*(x[1] - 12)**2 + x[2]**4 + 3*(x[3] - 11)**2 + 10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]
        return f_x

    @staticmethod
    def cec2006_g09_g1(x):  # restriccion 1 de desigualdad <= 0
        result = -127 + 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4]
        return result

    @staticmethod
    def cec2006_g09_g2(x):  # restriccion 2 de desigualdad <= 0
        result = -282 + 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4]
        return result

    @staticmethod
    def cec2006_g09_g3(x):  # restriccion 3 de desigualdad <= 0
        result = -196 + 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6]
        return result

    @staticmethod
    def cec2006_g09_g4(x):  # restriccion 4 de desigualdad <= 0
        result = 4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]
        return result
