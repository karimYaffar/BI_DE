import numpy as np
from .Problem import Problem, ProblemType

# Problema 01
class CEC2020_RC01(Problem):
    
    # WITH BOUNDS:
    
    # 0 ≤ x≤ 10, 0 ≤ x2 ≤ 200, 0 ≤ x3 ≤ 100, 0 ≤ x4 ≤ 200,
    # 1000 ≤ x5 ≤ 2000000, 0 ≤ x6 ≤ 600, 100 ≤ x7 ≤ 600, 100 ≤ x8 ≤ 600,
    # 100 ≤ x9 ≤ 900.
    
    SUPERIOR = np.array([10,200,100,200,2000000,600,600,600,900])
    INFERIOR = np.array([0,0,0,0,1000,0,100,100,100])

    def __init__(self):
        rest_h = [
            self.CEC2020_RC01_h1,
            self.CEC2020_RC01_h2,
            self.CEC2020_RC01_h3,
            self.CEC2020_RC01_h4,
            self.CEC2020_RC01_h5,
            self.CEC2020_RC01_h6,
            self.CEC2020_RC01_h7,
            self.CEC2020_RC01_h8,
        ]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, [],  rest_h)
    
    def fitness(self, individuo: np.array) -> float:
        f_x = 35 * individuo[0]**0.6 + 35 * individuo[1]**0.6
        return f_x

    @staticmethod
    def CEC2020_RC01_h1(x):
        return 200 * x[0] * x[3] - x[2]  # restriccion de igualdad = 0
    
    @staticmethod
    def CEC2020_RC01_h2(x):
        return 200 * x[1] * x[5] - x[4]  # restriccion 2 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC01_h3(x):
        return x[2] - 10000 * (x[6] - 100)  # restriccion 3 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC01_h4(x):
        return x[4] - 10000 * (300 - x[6])  # restriccion 4 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC01_h5(x):
        return x[2] - 10000 * (600 - x[7]) # restriccion 5 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC01_h6(x):
        return x[4] - 10000 * (900 - x[8]) # restriccion 6 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC01_h7(x):
        if x[7] - 100 <= 0 or 600 - x[6] <= 0:
            return 0
        return x[3] * np.log(x[7] - 100) - x[3] * np.log(600 - x[6]) - x[7] + x[6] + 500

    @staticmethod
    def CEC2020_RC01_h8(x):
        if x[8] - x[6] <= 0 or 600 <= 0:
            return 0 
        return x[5] * np.log(x[8] - x[6]) - x[5] * np.log(600) - x[8] + x[6] + 600

#********************************************************************************************************************************

#Problem 02

class CEC2020_RC02(Problem):
    
    # WITH BOUNDS:
    
    # 10**4 ≤ x≤ 81.9 × 10**4, 10**4 ≤ x2 ≤ 113.× 10**4, 10**4 ≤ x3 ≤ 205 × 10**4,
    # 0 ≤ x4, x5, x6 ≤ 5.074 × 10**−2, 100 ≤ x7 ≤ 200, 100 ≤ x8, x9, x10 ≤ 300
    # 100 ≤ x1≤ 400.

    
    INFERIOR = np.array([
        10**4,10**4,10**4,0,0,
        0,100,100,100,100,100
    ])
    SUPERIOR = np.array([
        81.9 * 10**4,113.* 10**4,205 * 10**4,5.074 * 10**-2,5.074 * 10**-2,
        5.074 * 10**-2,200,300,300,300,400
    ]) 
    
    def __init__(self):
        rest_h = [
            self.CEC2020_RC02_h1, self.CEC2020_RC02_h2, self.CEC2020_RC02_h3,
            self.CEC2020_RC02_h4, self.CEC2020_RC02_h5, self.CEC2020_RC02_h6,
            self.CEC2020_RC02_h7, self.CEC2020_RC02_h8, self.CEC2020_RC02_h9
        ]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, [],  rest_h)
    
    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = ( x[0] / (120 * x[3]) )**0.6 + ( x[1] / (80 * x[4]) )**0.6 + ( x[2] / (40 * x[5]) )**0.6
        return f_x
    
    @staticmethod
    def CEC2020_RC02_h1(x):
        return x[0] - 10**4 * (x[6] - 100)   # restriccion de igualdad = 0
    
    @staticmethod
    def CEC2020_RC02_h2(x):
        return x[1] - 10**4 * (x[7] - x[6])  # restriccion 2 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC02_h3(x):
        return x[2] - 10**4 * (500 - x[7])  # restriccion 3 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC02_h4(x):
        return x[0] - 10**4 * (300 - x[8])  # restriccion 4 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC02_h5(x):
        return x[1] - 10**4 * (400 - x[9])  # restriccion 5 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC02_h6(x):
        return x[3] - 10**4 * (600 - x[10])  # restriccion 6 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC02_h7(x):
        if x[8] - 100 <= 0 or 300 - x[6] <= 0:
            return 0
        return x[3] * np.log(x[8] - 100) - x[3] * np.log(300 - x[6]) - x[8] + x[6] + 400 # restriccion 7 de igualdad = 0
    
    @staticmethod              
    def CEC2020_RC02_h8(x):
        if x[9] - x[6] <= 0 or 400 - x[7] <= 0:
            return 0
        return x[4] * np.log(x[9] - x[6]) - x[4] * np.log(400 - x[7]) - x[9] + x[6] - x[7] + 400 #restriccion 8 de igualdad = 0

    @staticmethod
    def CEC2020_RC02_h9(x):
        if x[10] - x[7] <= 0:
            return 0
        return x[5] * np.log(x[10] - x[7]) - x[5] * np.log(100) - x[10] + x[7] + 100 # restriccion 9 de igualdad = 0

#********************************************************************************************************************************

#Problem 03

class CEC2020_RC03(Problem):
    
    # WITH BOUNDS:
    # 
    # 1000 ≤ x1 ≤ 2000, 0 ≤ x2 ≤ 100
    # 2000 ≤ x3 ≤ 4000, 0 ≤ x4 ≤ 100
    # 0 ≤ x5 ≤ 100, 0 ≤ x6 ≤ 20
    # 0 ≤ x7 ≤ 200.

    SUPERIOR = np.array([2000, 100, 4000, 100, 100, 20, 200]) 
    INFERIOR = np.array([1000, 0, 2000, 0, 0, 0, 0])
    
    def __init__(self):
        rest_h = [
            self.CEC2020_RC03_g1, self.CEC2020_RC03_g2, self.CEC2020_RC03_g3,
            self.CEC2020_RC03_g4, self.CEC2020_RC03_g5, self.CEC2020_RC03_g6,
            self.CEC2020_RC03_g7, self.CEC2020_RC03_g8, self.CEC2020_RC03_g9,
            self.CEC2020_RC03_g10, self.CEC2020_RC03_g11, self.CEC2020_RC03_g12,
            self.CEC2020_RC03_g13, self.CEC2020_RC03_g14
        ]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, [],  rest_h)
    
    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = 0.035 * x[0] * x[5] + 1.715 * x[0] + 10.0 * x[2] + 4.0565 * x[2] - 0.063 * x[2] * x[4]
        return f_x
    
    @staticmethod
    def CEC2020_RC03_g1(x):
        return 0.0059553571 * x[5]**2 * x[0] + 0.88392857 * x[2] - 0.1175625 * x[5] * x[0] - x[0]   # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g2(x):
        return 1.1088 * x[0] + 0.1303533 * x[0] * x[5] - 0.0066033 * x[0] * x[5]**2 - x[2]  # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g3(x):
        return  6.66173269 * x[5]**2 - 56.596669 * x[3] + 172.39878 * x[4] - 10000 - 191.20592 * x[5]   # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g4(x):
        return 1.08702 * x[5] - 0.03762 * x[5]**2 + 0.32175 * x[3] + 56.85075 - x[4]  # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g5(x):
        return 0.006198* x[6] * x[3] * x[2] + 2462.3121 * x[1] - 25.125634 * x[1] * x[3] - x[2] * x[3]  # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g6(x):
        return  161.18996 * x[2] * x[3] + 5000.0 * x[1] * x[3] - 489510.0 * x[1] - x[2] * x[3] * x[6]  # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g7(x):
        return 0.33 * x[6] + 44.333333 - x[4] # restriccion de desigualdad <= 0
    
    @staticmethod              
    def CEC2020_RC03_g8(x):
        return 0.022556 * x[4] - 1.0 - 0.007595 * x[6] #restriccion de desigualdad <= 0

    @staticmethod
    def CEC2020_RC03_g9(x):
        return 0.00061 * x[2] - 1.0 - 0.0005 * x[0]  # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g10(x):
        return 0.819672 * x[0] - x[2] + 0.819672 # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g11(x):
        return 24500.0 * x[1] - 250.0 * x[1] * x[3] - x[2] * x[3] # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g12(x):
        return 1020.4082 * x[3] * x[1] + 1.2244898 * x[2] * x[3] - 100000 * x[1] # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g13(x):
        return 6.25 * x[0] * x[5] + 6.25 * x[0] - 7.625 * x[2] - 100000 # restriccion de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC03_g14(x):
        return 1.22 * x[2] - x[5] * x[0] - x[0] + 1.0  # restriccion de desigualdad <= 0

#********************************************************************************************************************************

#Problem 04
class CEC2020_RC04(Problem):
    
    # WITH BOUNDS:
    
    # 0 ≤ x4, x3, x2, x1 ≤ 1,
    # 0.00001 ≤ x6, x5 ≤ 16.
    
    SUPERIOR = np.array([1, 1, 1, 1, 16, 16])
    INFERIOR = np.array([0, 0, 0, 0, 0.00001, 0.00001])
    
    k3 = 0.0391908
    k4 = 0.9 * k3
    k1 = 0.09755988
    k2 = 0.99 * k1

    def __init__(self):
        rest_h = [
            self.CEC2020_RC04_h1, self.CEC2020_RC04_h2, self.CEC2020_RC04_h3, self.CEC2020_RC04_h4, 
        ]
        rest_g = [
            self.CEC2020_RC04_g1
        ]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)
    
    def fitness(self, individuo: np.array) -> float:
        x = np.copy(individuo)
        f_x = x[3]
        return f_x

    @staticmethod
    def CEC2020_RC04_h1(x):
        k1 = CEC2020_RC04.k1
        return k1 * x[4] * x[1] + x[0] - 1  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC04_h2(x):
        k3 = CEC2020_RC04.k3
        return k3 * x[4] * x[2] + x[2] + x[0] - 1  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC04_h3(x):
        k2 = CEC2020_RC04.k2
        return k2 * x[5] * x[1] - x[0] + x[1]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC04_h4(x):
        k4 = CEC2020_RC04.k4
        return k4 * x[5] * x[3] + x[1] - x[0] + x[3] - x[2]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC04_g1(x):
        return x[4]**0.5 + x[5]**0.5 - 4  # restricción de desigualdad <= 0


#********************************************************************************************************************************

#Problem 05

class CEC2020_RC05(Problem):
    
    # WITH BOUNDS:
    
    # 0 ≤ x1, x3, x4, x5, x6, x8 ≤ 100, 0 ≤ x2, x7, x9 ≤ 200.
    
    SUPERIOR = np.array([100,200,100,100,100,100,200,100,200])
    INFERIOR = np.zeros(9)

    def __init__(self):
        rest_h = [
            self.CEC2020_RC05_h1, self.CEC2020_RC05_h2, self.CEC2020_RC05_h3, self.CEC2020_RC05_h4, 
        ]
        rest_g = [
            self.CEC2020_RC05_g1, self.CEC2020_RC05_g2
        ]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g,  rest_h)
    
    def fitness(self, individuo: np.array) -> float:
        x = np.copy(individuo)
        f_x = 9 * x[0] + 15 * x[1] - 6 * x[2] - 16 * x[3] - 10 * (x[4] + x[5])
        return f_x

    @staticmethod
    def CEC2020_RC05_h1(x):
        return x[6] + x[7] - x[3] - x[2]  # restriccion de igualdad = 0
    
    @staticmethod
    def CEC2020_RC05_h2(x):
        return x[0] - x[4] - x[7]  # restriccion 2 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC05_h3(x):
        return x[1] - x[5] - x[6]  # restriccion 3 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC05_h4(x):
        return x[8] * x[6] + x[8] * x[7] - 3 * x[2] - x[3]   # restriccion 4 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC05_g1(x):
        return x[8] * x[6] + 2 * x[4] - 2.5 * x[0]  # restriccion 5 de igualdad = 0
    
    @staticmethod
    def CEC2020_RC05_g2(x):
        return x[8] * x[7] + 2 * x[5] - 1.5 * x[1] # restriccion 6 de igualdad = 0
    
#********************************************************************************************************************************

#Problem 06

class CEC2020_RC06(Problem):
    
    SUPERIOR = np.array([
        90,  150,  90, 150,   90,  90, 150,  90,  90,  90, 150, 150, 90,  90,  
        150, 90,  150,  90,  150,  90,   1, 1.2,   1,   1,   1, 0.5,  1,   1,
        0.5, 0.5, 0.5, 1.2,  0.5, 1.2, 1.2, 0.5, 1.2, 1.2
    ])
    
    INFERIOR = np.zeros(38)
    
    def __init__(self):
        rest_h = [
            self.CEC2020_RC06_h1, self.CEC2020_RC06_h2, self.CEC2020_RC06_h3, self.CEC2020_RC06_h4,
            self.CEC2020_RC06_h5, self.CEC2020_RC06_h6, self.CEC2020_RC06_h7, self.CEC2020_RC06_h8,
            self.CEC2020_RC06_h9, self.CEC2020_RC06_h10, self.CEC2020_RC06_h11, self.CEC2020_RC06_h12,
            self.CEC2020_RC06_h13, self.CEC2020_RC06_h14, self.CEC2020_RC06_h15, self.CEC2020_RC06_h16,
            self.CEC2020_RC06_h17, self.CEC2020_RC06_h18, self.CEC2020_RC06_h19, self.CEC2020_RC06_h20,
            self.CEC2020_RC06_h21, self.CEC2020_RC06_h22, self.CEC2020_RC06_h23, self.CEC2020_RC06_h24,
            self.CEC2020_RC06_h25, self.CEC2020_RC06_h26, self.CEC2020_RC06_h27, self.CEC2020_RC06_h28,
            self.CEC2020_RC06_h27, self.CEC2020_RC06_h30, self.CEC2020_RC06_h31, self.CEC2020_RC06_h32,
        ]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, [],  rest_h)
    
    def fitness(self, individuo: np.array) -> float:
        x = np.copy(individuo)
        f_x = 0.9979 + 0.00432 * x[4] + 0.01517 * x[12]
        return f_x

    @staticmethod
    def CEC2020_RC06_h1(x):
        return x[3] + x[2] + x[1] + x[0] - 300  # restricción de igualdad = 300

    @staticmethod
    def CEC2020_RC06_h2(x):
        return x[5] - x[7] - x[6]   # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h3(x):
        return x[8] - x[10] - x[9] - x[11]  # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h4(x):
        return x[13] - x[15] - x[16] - x[14]    # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h5(x):
        return x[17] - x[19] - x[18]    # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h6(x):
        return x[4] * x[20] - x[5] * x[21] - x[8] * x[22]   # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h7(x):
        return x[4] * x[23] - x[5] * x[24] - x[8] * x[25]   # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h8(x):
        return x[4] * x[26] - x[5] * x[27] - x[8] * x[28]   # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h9(x):
        return x[12] * x[29] - x[13] * x[30] - x[17] * x[31]    # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h10(x):
        return x[12] * x[32] - x[13] * x[33] - x[17] * x[34]    # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h11(x):
        return x[12] * x[35] - x[13] * x[36] - x[17] * x[34]    # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h12(x):
        return 0.333 * x[0] + x[14] * x[30] - x[4] * x[20]  # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h13(x):
        return 0.333 * x[0] + x[14] * x[33] - x[4] * x[23]  # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h14(x):
        return 0.333 * x[0] + x[14] * x[36] - x[4] * x[26]  # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h15(x):
        return 0.333 * x[1] + x[9] * x[22] - x[12] * x[29]  # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h16(x):
        return 0.333 * x[1] + x[9] * x[25] - x[12] * x[32]  # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h17(x):
        return 0.333 * x[1] + x[9] * x[28] - x[12] * x[35]  # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC06_h18(x):
        return 0.333 * x[2] + x[6] * x[21] + x[10] * x[22] + x[15] * x[30] + x[18] * x[31] - 30 # restricción de igualdad = 30

    @staticmethod
    def CEC2020_RC06_h19(x):
        return 0.333 * x[2] + x[6] * x[24] + x[10] * x[25] + x[15] * x[33] + x[18] * x[34] - 50 # restricción de igualdad = 50

    @staticmethod
    def CEC2020_RC06_h20(x):
        return 0.333 * x[2] + x[6] * x[27] + x[10] * x[28] + x[15] * x[36] + x[18] * x[37] - 30 # restricción de igualdad = 30

    @staticmethod
    def CEC2020_RC06_h21(x):
        return x[20] + x[23] + x[26] - 1    # restricción de igualdad = 1

    @staticmethod
    def CEC2020_RC06_h22(x):
        return x[21] + x[24] + x[27] - 1    # restricción de igualdad = 1

    @staticmethod
    def CEC2020_RC06_h23(x):
        return x[22] + x[25] + x[28] - 1    # restricción de igualdad = 1

    @staticmethod
    def CEC2020_RC06_h24(x):
        return x[29] + x[32] + x[35] - 1    # restricción de igualdad = 1

    @staticmethod
    def CEC2020_RC06_h25(x):
        return x[30] + x[33] + x[36] - 1    # restricción de igualdad = 1

    @staticmethod
    def CEC2020_RC06_h26(x):
        return x[31] + x[34] + x[37] - 1    # restricción de igualdad = 1

    @staticmethod
    def CEC2020_RC06_h27(x):
        return x[25]    

    @staticmethod
    def CEC2020_RC06_h28(x):
        return x[27]

    @staticmethod
    def CEC2020_RC06_h29(x):
        return x[22]

    @staticmethod
    def CEC2020_RC06_h30(x):
        return x[36]

    @staticmethod
    def CEC2020_RC06_h31(x):
        return x[31]

    @staticmethod
    def CEC2020_RC06_h32(x):
        return x[34]

#********************************************************************************************************************************

#Problem 07

class CEC2020_RC07(Problem):
    
    # with bounds :
    # 0 ≤ x1, ..., x20 ≤ 150; 0 ≤ x25, x27, x32, x35, x37, x29 ≤ 30;
    # 0 ≤ x21, x22, x23, x30, x33, x34, x36, x38, x39, x40, x42, x43, x44, x45, ≤ 1,
    # 0 ≤ x46, x47, x48 ≤ 1,
    # 0.85 ≤ x24, x26, x28, x3≤ 1
    
    SUPERIOR = np.array([
        150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
        150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
        1, 1, 1, 1, 30, 1, 1, 30, 1, 1,
        1, 1, 1, 30, 1, 1, 1, 1, 1, 1,
        30, 30, 30, 1, 1, 1, 1, 1
    ])
    
    INFERIOR = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0.85, 0, 0.85, 0, 0.85, 0,
        0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ])
    
    def __init__(self):
        
        # Coeficientes dados en la tabla
        self.c = np.array([
            [0.23947, 0.758535],
            [-0.0139904, -0.0661588],
            [0.0093514, 0.0338147],
            [0.0077508, 0.0373349],
            [-0.0057719, 0.00163711],
            [0.0042656, 0.0286996]
        ])
        
        rest_h = [
            self.CEC2020_RC07_h1, self.CEC2020_RC07_h2, self.CEC2020_RC07_h3, self.CEC2020_RC07_h4,
            self.CEC2020_RC07_h5, self.CEC2020_RC07_h6, self.CEC2020_RC07_h7, self.CEC2020_RC07_h8,
            self.CEC2020_RC07_h9, self.CEC2020_RC07_h10, self.CEC2020_RC07_h11, self.CEC2020_RC07_h12,
            self.CEC2020_RC07_h13, self.CEC2020_RC07_h14, self.CEC2020_RC07_h15, self.CEC2020_RC07_h16,
            self.CEC2020_RC07_h17, self.CEC2020_RC07_h18, self.CEC2020_RC07_h19, self.CEC2020_RC07_h20,
            self.CEC2020_RC07_h21, self.CEC2020_RC07_h22, self.CEC2020_RC07_h23, self.CEC2020_RC07_h24,
            self.CEC2020_RC07_h25, self.CEC2020_RC07_h26, self.CEC2020_RC07_h27, self.CEC2020_RC07_h28,
            self.CEC2020_RC07_h29, self.CEC2020_RC07_h30, self.CEC2020_RC07_h31, self.CEC2020_RC07_h32,
            self.CEC2020_RC07_h33, self.CEC2020_RC07_h34, self.CEC2020_RC07_h35, self.CEC2020_RC07_h36,
            self.CEC2020_RC07_h37, self.CEC2020_RC07_h38
        ]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, [],  rest_h)


    def fitness(self, x: np.array) -> float:
        c = self.c
        term1 = c[0, 0]
        term2 = (c[1, 0] + c[2, 0] * x[23] + c[3, 0] * x[36] + c[4, 0] * x[51] + c[5, 0] * x[43] + c[5, 0] * x[11]) * x[4]
        term3 = c[0, 1]
        term4 = (c[1, 1] + c[2, 1] * x[24] + c[3, 1] * x[29] + c[4, 1] * x[36] + c[5, 1] * x[37] + c[5, 1] * x[11]) * x[12]
        return term1 + term2 + term3 + term4

    @staticmethod
    def CEC2020_RC07_h1(x):
        return x[3] + x[2] + x[1] + x[0] - 300  # restricción de igualdad = 300
    
    @staticmethod
    def CEC2020_RC07_h2(x):
        return x[5] - x[7] - x[6]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h3(x):
        return x[8] - x[11] - x[9] - x[10]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h4(x):
        return x[13] - x[16] - x[14] - x[15]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h5(x):
        return x[17] - x[19] - x[18]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h6(x):
        return x[5] * x[20] - x[23] * x[24]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h7(x):
        return x[13] * x[21] - x[25] * x[26]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h8(x):
        return x[8] * x[22] - x[27] * x[28]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h9(x):
        return x[17] * x[29] - x[30] * x[31]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h10(x):
        return x[24] - x[4] * x[32]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h11(x):
        return x[28] - x[4] * x[33]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h12(x):
        return x[34] - x[4] * x[35]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h13(x):
        return x[36] - x[12] * x[37]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h14(x):
        return x[26] - x[12] * x[38]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h15(x):
        return x[31] - x[12] * x[39]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h16(x):
        return x[24] - x[5] * x[20] - x[8] * x[40]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h17(x):
        return x[28] - x[5] * x[41] - x[8] * x[22]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h18(x):
        return x[34] - x[5] * x[42] - x[8] * x[43]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h19(x):
        return x[36] - x[13] * x[44] - x[17] * x[45]  # restricción de igualdad = 0

    @staticmethod
    def CEC2020_RC07_h20(x):
        return x[26] - x[13] * x[21] - x[17] * x[46]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h21(x):
        return x[31] - x[13] * x[47] - x[17] * x[29]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h22(x):
        return 0.333 * x[0] + x[14] * x[44] - x[24]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h23(x):
        return 0.333 * x[0] + x[14] * x[21] - x[28]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h24(x):
        return 0.333 * x[0] + x[14] * x[47] - x[34]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h25(x):
        return 0.333 * x[1] + x[9] * x[40] - x[36]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h26(x):
        return 0.333 * x[1] + x[9] * x[22] - x[26]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h27(x):
        return 0.333 * x[1] + x[9] * x[43] - x[31]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h28(x):
        return 0.333 * x[2] + x[6] * x[20] + x[10] * x[40] + x[15] * x[44] + x[18] * x[45] - 30  # restricción de igualdad = 30
    
    @staticmethod
    def CEC2020_RC07_h29(x):
        return 0.333 * x[2] + x[6] * x[41] + x[10] * x[22] + x[15] * x[21] + x[18] * x[46] - 50  # restricción de igualdad = 50
    
    @staticmethod
    def CEC2020_RC07_h30(x):
        return 0.333 * x[2] + x[6] * x[42] + x[10] * x[43] + x[15] * x[47] + x[18] * x[29] - 30  # restricción de igualdad = 30
    
    @staticmethod
    def CEC2020_RC07_h31(x):
        return x[32] + x[33] + x[35] - 1  # restricción de igualdad = 1
    
    @staticmethod
    def CEC2020_RC07_h32(x):
        return x[20] + x[41] + x[42] - 1  # restricción de igualdad = 1
    
    @staticmethod
    def CEC2020_RC07_h33(x):
        return x[40] + x[22] + x[43] - 1  # restricción de igualdad = 1
    
    @staticmethod
    def CEC2020_RC07_h34(x):
        return x[37] + x[38] + x[39] - 1  # restricción de igualdad = 1
    
    @staticmethod
    def CEC2020_RC07_h35(x):
        return x[44] + x[21] + x[47] - 1  # restricción de igualdad = 1
    
    @staticmethod
    def CEC2020_RC07_h36(x):
        return x[45] + x[46] + x[29] - 1  # restricción de igualdad = 1
    
    @staticmethod
    def CEC2020_RC07_h37(x):
        return x[42]  # restricción de igualdad = 0
    
    @staticmethod
    def CEC2020_RC07_h38(x):
        return x[45]  # restricción de igualdad = 0

#********************************************************************************************************************************

#Problem 08

import numpy as np

class CEC2020_RC08(Problem):
    
    # WITH BOUNDS:
    # 0 ≤ x1 ≤ 1.6
    # x2 ∈ {0, 1}
    
    SUPERIOR = np.array([1.6, 1])
    INFERIOR = np.array([0, 0])
    
    def __init__(self):
        rest_g = [
            self.CEC2020_RC08_g1, self.CEC2020_RC08_g2
        ]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, [])
    
    def fitness(self, individuo: np.array) -> float:
        x = np.copy(individuo)
        f_x = x[1] + 2 * x[0]
        return f_x

    @staticmethod
    def CEC2020_RC08_g1(x):
        return -x[0]**2 - x[1] + 1.25  # restricción de desigualdad <= 0
    
    @staticmethod
    def CEC2020_RC08_g2(x):
        return x[0] + x[1] - 1.6  # restricción de desigualdad <= 0


#********************************************************************************************************************************

#Problem 09
