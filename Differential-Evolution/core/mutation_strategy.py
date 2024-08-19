import numpy as np
from utils.constants import GENERATIONS, SIZE_POPULATION
from functools import lru_cache

class MutationStrategies:
    def __init__(self, population, scale, objective_function):
        self.population = population
        self.scale = scale
        self.objective_function = objective_function
        
    def _best1(self, samples):
        r0, r1 = samples[:2]
        return self.population[0] + self.scale * (self.population[r0] - self.population[r1])

    def _rand1(self, samples):
        r0, r1, r2 = samples[:3]
        return self.population[r0] + self.scale * (self.population[r1] - self.population[r2])
    
    def _currenttobest1(self, candidate, samples):
        r0, r1 = samples[:2]
        return (self.population[candidate] + self.scale *
                (self.population[0] - self.population[candidate] +
                 self.population[r0] - self.population[r1]))

    def _best2(self, samples):
        r0, r1, r2, r3 = samples[:4]
        return (self.population[0] + self.scale *
                (self.population[r0] + self.population[r1] -
                 self.population[r2] - self.population[r3]))

    def _rand2(self, samples):
        r0, r1, r2, r3, r4 = samples
        return (self.population[r0] + self.scale *
                (self.population[r1] + self.population[r2] -
                 self.population[r3] - self.population[r4]))
    ########
    def _rand_to_best1(self, samples, bprime):
        r1, r2, r3, r4 = samples[:4]
        mutado = self.population[r1] + self.scale * (bprime - self.population[r2]) + self.scale * (self.population[r3] - self.population[r4])
        return mutado
    ########
    def _randtobest1(self, samples):
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] - self.population[r2])
        return bprime

    def _currenttorand1(self, candidate, samples):
        r0, r1, r2 = samples[:3]
        x_i = np.copy(self.population[candidate])
        x_r = self.population[r0]
        x_s = self.population[r1]
        
        bprime = x_i + self.scale * (self.population[0] - x_i) + self.scale * (x_r - x_s)
        return bprime

    def combined_rand1_best2(self, probability_rand1):
        if np.random.rand() < probability_rand1:
            samples = np.random.choice(len(self.population), size=3, replace=False)
            return self._rand1(samples)
        else:
            samples = np.random.choice(len(self.population), size=6, replace=False)
            return self._best2(samples)



    def _rand3(self, samples):
        r0, r1, r2, r3, r4, r5 = samples[:6]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2] +
                 self.population[r3] - self.population[r4] +
                 self.population[r5]))

    def _best3(self, samples):
        r0, r1, r2, r3, r4, r5 = samples[:6]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1] +
                 self.population[r2] - self.population[r3] +
                 self.population[r4] - self.population[r5]))

    def _rand_to_current2(self, candidate, samples):
        r0, r1 = samples[:2]
        return (self.population[candidate] + self.scale *
                (self.population[r0] - self.population[candidate] +
                 self.population[r1]))

    def _rand_to_best_and_current2(self, candidate, samples):
        r0, r1, r2, r3 = samples[:4]
        return (self.population[candidate] + self.scale *
                (self.population[0] - self.population[candidate] +
                 self.population[r0] - self.population[r1] +
                 self.population[r2] - self.population[r3]))
                                                
    """ @lru_cache(maxsize=None)
    def cached_objective_function(self, idx):
        return self.objective_function(self.population[idx])

    def _select_pbest(self):
        population_size = len(self.population)
        num_top_individuals = int(self.p * population_size)
        if num_top_individuals == 0:
            num_top_individuals = 1
        fitness_values = [self.cached_objective_function(i) for i in range(population_size)]
        top_indices = np.argpartition(fitness_values, num_top_individuals)[:num_top_individuals]
        return self.population[np.random.choice(top_indices)]

    def _currenttopbest1(self, candidate, samples):
        r1, r2 = samples[:2]
        x_i = self.population[candidate]
        x_pbest = self._select_pbest()
        x_r1 = self.population[r1]
        x_r2 = self.population[r2]
        
        mutant = x_i + self.scale * (x_pbest - x_i) + self.scale * (x_r1 - x_r2)
        return mutant
 """
    
    def _adaptive_rand_elite(self, generation):
        # Definir parámetros
        F_min = 0.5
        F_max = 0.9
        beta = 0.5
        G = GENERATIONS

        # Calcula F adaptativor
        F_adaptive = F_min + (F_max - F_min) * (1 - generation / G)

        # Seleccionar individuos
        elite_idx = np.argsort([self.objective_function(ind) for ind in self.population])[:int(0.1 * SIZE_POPULATION)]
        elite = self.population[np.random.choice(elite_idx)]
        
        samples = np.random.choice(SIZE_POPULATION, 3, replace=False)
        r1, r2, r3 = samples[:3]

        mutant = elite + F_adaptive * (self.population[r1] - self.population[r2]) + beta * (self.population[0] - self.population[r3])
        return mutant
