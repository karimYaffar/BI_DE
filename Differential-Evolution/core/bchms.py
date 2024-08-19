import numpy as np

class BCHM:
    
    @staticmethod
    def reflection(upper, lower, x):
        range_width = upper - lower
        new_x = np.copy(x)

        for j in range(len(x)):
            if new_x[j] < lower[j]:
                new_x[j] = lower[j] + (lower[j] - new_x[j]) % range_width[j]
            elif new_x[j] > upper[j]:
                new_x[j] = upper[j] - (new_x[j] - upper[j]) % range_width[j]

        return new_x
    
    @staticmethod
    def boundary(x, lower_bound, upper_bound):
        x_projected = np.copy(x)
        x_projected = np.where(x_projected < lower_bound, lower_bound, x_projected)
        x_projected = np.where(x_projected > upper_bound, upper_bound, x_projected)
        return x_projected
    
    @staticmethod
    def random_component(upper, lower, x):
        # Crear una máscara para identificar los elementos fuera de los límites
        mask_lower = x < lower
        mask_upper = x > upper
        # Crear una copia de x para no modificar el original
        corrected_x = np.copy(x)        
        # Asignar valores aleatorios dentro de los límites solo a los elementos fuera de los límites
        corrected_x[mask_lower] = np.random.uniform(lower[mask_lower], upper[mask_lower])
        corrected_x[mask_upper] = np.random.uniform(lower[mask_upper], upper[mask_upper])        
        return corrected_x
    
    @staticmethod
    def random_all(upper, lower, x):
        return np.random.uniform(lower, upper, size=x.shape)

    @staticmethod
    def wrapping(upper, lower, x):
        range_width = abs(upper - lower)
        new_x = np.copy(x)

        for i in range(len(x)):
            if new_x[i] < lower[i]:
                new_x[i] = upper[i] - (lower[i] - new_x[i]) % range_width[i]
            elif new_x[i] > upper[i]:
                new_x[i] = lower[i] + (new_x[i] - upper[i]) % range_width[i]

        return new_x

    @staticmethod
    def evolutionary(z, lb, ub, best_solution):
        a = np.random.rand()
        b = np.random.rand()

        x = np.copy(z)

        x[z < lb] = a * lb[z < lb] + (1 - a) * best_solution[z < lb]
        x[z > ub] = b * ub[z > ub] + (1 - b) * best_solution[z > ub]

        return x
    
    @staticmethod
    def evo_cen(x, population, lower, upper, SIS, SFS, gbest_individual, K=1):
        x_evo = BCHM.evolutionary(x, lower, upper, gbest_individual)
        
        return BCHM.centroid(x_evo, population, lower, upper, SFS, SIS, gbest_individual, K)
    
    @staticmethod
    def centroid(X, population, lower, upper, SFS, SIS, gbest_individual, K=1):
        if len(SFS) > 0 and np.random.rand() > 0.5:
            random_position_index = np.random.choice(SFS) # Elegir una posición aleatoria de SFS
            We = population[random_position_index] # Obtener el individuo de la población en esa posición            
        else:
            if len(SIS) > 0:
                We = gbest_individual
            else:                
                We = population[np.random.randint(population.shape[0])]

        Wr = [BCHM.random_component(upper, lower, X) for _ in range(K)]

        Wr = np.array(Wr)

        sum_components = We + Wr.sum(axis=0)

        result = sum_components / (K + 1)

        return result
    

    ######################################################

    @staticmethod
    def evo_cen_component(x, lower, upper, gbest_individual):
        x_evo = BCHM.evolutionary(x, lower, upper, gbest_individual)
        
        # Generar un nuevo vector aleatorio para componentes fuera de los límites
        x_random_component = BCHM.random_component(upper, lower, x)
        
        # Calcular el centroide de los tres vectores
        result = (x_evo + gbest_individual + x_random_component) / 3
        
        return result
    
    @staticmethod
    def evo_cen_all(x, lower, upper, gbest_individual):
        x_evo = BCHM.evolutionary(x, lower, upper, gbest_individual)
    
        # Generar un nuevo vector aleatorio dentro de los límites usando random_all
        x_random_all = BCHM.random_all(upper, lower, x)
    
        # Calcular el centroide de los tres vectores
        result = (x_evo + gbest_individual + x_random_all) / 3
    
        return result
    
    ######################################################
    
    @staticmethod
    def vector_wise_correction(x, upper, lower, method="midpoint"):
        if method == "midpoint":
            R = (lower + upper) / 2
        else:
            raise ValueError(f"Método para calcular R no soportado: {method}")
        
        
        alpha = np.min(
            [
                np.where(x < lower, (R - lower) / (R - x), 1),
                np.where(x > upper, (upper - R) / (x - R), 1),
            ]
        )
        return alpha * x + (1 - alpha) * R
    
    @staticmethod
    def beta(individual, lower_bound, upper_bound, population):
        corrected_individual = np.copy(individual)
        population_mean = np.mean(population, axis=0)
        population_var = np.var(population, axis=0)
        
        for i in range(len(individual)):
            m_i = (population_mean[i] - lower_bound[i]) / (upper_bound[i] - lower_bound[i])
            v_i = population_var[i] / (upper_bound[i] - lower_bound[i])**2
            
            if m_i == 0:
                m_i = 0.1
            elif m_i == 1:
                m_i = 0.9
            
            if v_i == 0:
                v_i = 1e-6
            
            alpha_i = m_i * ((m_i * (1 - m_i) / v_i) - 1)
            beta_i = alpha_i * (1 - m_i) / m_i
            
            if alpha_i <= 0 or beta_i <= 0:
                alpha_i = beta_i = 1  
            
            corrected_value = np.random.beta(alpha_i, beta_i) * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
            corrected_individual[i] = corrected_value
        
        return corrected_individual
    
    
    
    def evo_beta_cen(x, population, lower, upper, SIS, SFS, gbest_individual, K=1):
        x_evo = BCHM.evolutionary(x, lower, upper, gbest_individual)
        
        x_beta = BCHM.beta(x_evo, lower, upper, population)
        
        return BCHM.centroid(x_beta, population, lower, upper, SFS, SIS, gbest_individual)
    
    @staticmethod
    def diversity_threshold(population, gbest_individual):
        # Calcular la distancia media de los individuos en la población respecto a gbest_individual
        distances = np.linalg.norm(population - gbest_individual, axis=1)  # Calcula la distancia Euclidiana por filas
        mean_distance = np.mean(distances)  # Calcular la distancia media
        
        # Puedes ajustar el umbral según tus necesidades específicas, por ejemplo, multiplicando por un factor
        threshold = 1.5 * mean_distance  # Por ejemplo, multiplicar por 1.5
        
        return threshold
    
    @staticmethod
    def diversity(x, population, lower, upper, SIS, SFS, gbest_individual, K=1):
        if np.random.rand() < 0.7:
            # Aplicar la corrección centrada
            return BCHM.evo_beta_cen(x, population, lower, upper, SIS, SFS, gbest_individual)
        else:    
            # print("Menos Diversidad")
            return BCHM.evo_cen(x, population, lower, upper, SIS, SFS, gbest_individual)