from typing import Callable, Tuple, List
from .algorithm import Algorithm
from .constraints_functions import ConstriantsFunctionsHandler
from utils.constants import SIZE_POPULATION, GENERATIONS
""" from utils.check_pause import check_for_pause """
from tqdm import tqdm
from .mutation_strategy import MutationStrategies
import numpy as np
import time


class Differential_Evolution(Algorithm):
    def __init__(
        self,
        objective_function: Callable,
        constraints_functions: Callable,
        bounds_constraints: Callable,
        bounds: Tuple[List, List] = ([], []),
        g_functions: List[Callable] = [],
        h_functions: List[Callable] = [],
        F: float = 0.7,
        CR: float = 0.9,
        strategy: str = "rand1",
        centroid: bool = False,
        beta: bool = False,
        evolutionary: bool = False,
        res_and_rand: bool = False,
        
        
    ):

        self.F = F
        self.CR = CR
        self.upper, self.lower = bounds
        self.g_functions = g_functions
        self.h_functions = h_functions
        self.solutions_generate = []

        self.strategy = strategy
        self.centroid = centroid
        self.beta = beta
        self.evolutionary = evolutionary
        self.res_and_rand = res_and_rand

        # Lists to store gbest values for plotting convergence
        self.gbest_fitness_list = []
        self.gbest_violations_list = []

        self.population = self.generate(self.upper, self.lower)
        self.fitness = np.zeros(SIZE_POPULATION)
        self.violations = np.zeros(SIZE_POPULATION)
        self.objective_function = objective_function
        self.constraints_functions = constraints_functions
        self.SFS = []
        self.SIS = []
        self.bounds_constraints = bounds_constraints
        self.best_fitness = []
        self.best_violations = []
        self.compute_fitness_and_violations()
        

        self.get_gbest_pobulation_zero()

 
        self.mutation_strategies = MutationStrategies(self.population, self.F, self.objective_function)  # Pasar p aquí

        # Variable to store the generation of the first feasible solution
        self.first_feasible_generation = None

    def compute_fitness_and_violations(self):
        for index, individual in enumerate(self.population):
            fitness = self.objective_function(individual)
            self.fitness[index] = fitness

            total_de_violaciones = ConstriantsFunctionsHandler.sum_of_violations(
                self.g_functions, self.h_functions, individual
            )
            self.violations[index] = total_de_violaciones

    def _compute_SFS_SIS(self):
        self.SFS = np.where(self.violations == 0)[0]
        self.SIS = np.where(self.violations > 0)[0]
            
    def mutation_operator(self, idx, generation):
        samples = np.random.choice(SIZE_POPULATION, 5, replace=False)
        if self.strategy == 'combined_60%':
            return self.mutation_strategies.combined_rand1_best2(0.6)
        elif self.strategy == 'DE-rand-1-bin':
            return self.mutation_strategies._rand1(samples)
        elif self.strategy == 'best1':
            return self.mutation_strategies._best1(samples)
        elif self.strategy == 'rand2':
            return self.mutation_strategies._rand2(samples)
        elif self.strategy == 'rand2':
            return self.mutation_strategies._rand2(samples)
        elif self.strategy == 'randtobest1':
            return self.mutation_strategies._randtobest1(samples)
        elif self.strategy == 'DE-ctb-1-bin':
            return self.mutation_strategies._currenttobest1(idx, samples)
        elif self.strategy == 'best2':
            return self.mutation_strategies._best2(samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
    def res_and_ran_mutation(self, i):

        F = 0.8  # Factor de escala para la mutación
        NP = len(self.population)
        D = len(self.population[0])
        
        NoRes = 0
        valid = False

        while NoRes <= 3 * D and not valid:
            r1, r2, r3 = np.random.choice([idx for idx in range(NP) if idx != i], 3, replace=False)
            V = self.population[r1] + F * (self.population[r2] - self.population[r3])
            
            # Verifica si el vector mutante está dentro de los límites
            if np.all([self.lower[j] <= V[j] <= self.upper[j] for j in range(D)]):
                valid = True
            else:
                NoRes += 1
        
        # Si no se obtuvo un vector válido después del máximo de remuestreos
        if not valid:
            V = np.array([np.random.uniform(self.lower[j], self.upper[j]) if V[j] < self.lower[j] or V[j] > self.upper[j] else V[j] for j in range(D)])
        
        return V
        

    def crossover_operator(self, target, mutant):
        dimensions = len(target)
        trial = np.copy(target)
        j_rand = np.random.randint(dimensions)

        prob_crossover = np.random.rand(dimensions) < self.CR

        trial[prob_crossover | (np.arange(dimensions) == j_rand)] = mutant[
            prob_crossover | (np.arange(dimensions) == j_rand)
        ]

        return trial

    def selection_operator(self, idx, trial):
        trial_fitness = self.objective_function(trial)
        trial_violations = ConstriantsFunctionsHandler.sum_of_violations(
            self.g_functions, self.h_functions, trial
        )

        current_fitness = self.fitness[idx]
        current_violations = self.violations[idx]

        if not self.constraints_functions(
            current_fitness, current_violations, trial_fitness, trial_violations
        ):
            self.fitness[idx] = trial_fitness
            self.violations[idx] = trial_violations
            self.population[idx] = trial

    def get_gbest_pobulation_zero(self):
        self.position_initial = 0

        self.gbest_fitness = self.fitness[self.position_initial]
        self.gbest_violation = self.violations[self.position_initial]
        self.gbest_individual = self.population[self.position_initial]

        self.update_position_gbest_population()

    def update_position_gbest_population(self):
        for idx in range(SIZE_POPULATION):
            current_fitness = self.fitness[idx]
            current_violation = self.violations[idx]

            if not self.constraints_functions(
                self.gbest_fitness,
                self.gbest_violation,
                current_fitness,
                current_violation,
            ):
                self.gbest_fitness = current_fitness
                self.gbest_violation = current_violation
                self.gbest_individual = self.population[idx]

                # Store gbest values for plotting
                self.gbest_fitness_list.append(self.gbest_fitness)
                self.gbest_violations_list.append(self.gbest_violation)

                self.best_fitness = self.gbest_fitness
                self.best_violations = self.gbest_violation

    def report(self):
        start_time = time.time()

        # Calcular estadísticas
        mean_fitness = np.mean(self.gbest_fitness_list)
        std_fitness = np.std(self.gbest_fitness_list)
        mean_violations = np.mean(self.gbest_violations_list)
        std_violations = np.std(self.gbest_violations_list)

        end_time = time.time()
        execution_time = end_time - start_time

        print("================================")
        print("Solución Óptima")
        print("Individuo:", self.gbest_individual)
        print("Aptitud (Fitness):", self.gbest_fitness)
        print("Num Violaciones:", self.gbest_violation)
        print("================================")
        print("Estadísticas de Convergencia")
        print(f"Media de Fitness: {mean_fitness}")
        print(f"Desviación Estándar de Fitness: {std_fitness}")
        print(f"Media de Violaciones: {mean_violations}")
        print(f"Desviación Estándar de Violaciones: {std_violations}")
        print(f"Tiempo de Ejecución del Reporte: {execution_time} segundos")
        print("================================")

  

    
    def evolution(self, verbose: bool = True):
        for gen in tqdm(range(GENERATIONS), desc="Evolucionando"):
           self._compute_SFS_SIS()
           for i in range(SIZE_POPULATION):
                objective = self.population[i]
                mutant = self.mutation_operator(i, gen)  # Pasando 'generation' aquí
                trial = self.crossover_operator(objective, mutant)
                if not self.isValid(self.upper, self.lower, trial):
                    if self.centroid:
                        trial = self.bounds_constraints(
                            trial,
                            self.population,
                            self.lower,
                            self.upper, 
                            self.SFS ,
                            self.SIS , 
                            self.gbest_individual
                                
                        )
                    else:
                        trial = self.bounds_constraints(self.upper, self.lower, trial)
                self.selection_operator(i, trial)

                # Verificar si es el primer individuo factible
                if self.first_feasible_generation is None and self.violations[i] == 0:
                    self.first_feasible_generation = gen

        self.update_position_gbest_population()

    
    

        if verbose:
            self.report()