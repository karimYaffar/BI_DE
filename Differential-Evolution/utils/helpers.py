import gc
import matplotlib.pyplot as plt
import pandas as pd
import os
import string
import random
from core.differential_evolution import Differential_Evolution
from core.constraints_functions import ConstriantsFunctionsHandler
from utils.constants import EXECUTIONS
from functions.cec2017problems import *

DIRECTORY = "report/cec2006"
PROBLEM = "CEC2006"

def execute_algorithm(problem_name, problem_class, constraint_name, bounds):
    problema = problem_class()
    fitness_data = []
    violations_data = []
    convergence_fitness_data = []
    convergence_violations_data = []

    for _ in range(EXECUTIONS):
        print(f"Ejecucion {_ + 1} para problema {problem_name} con el metodo {constraint_name}:")
        try:
            algorithm = Differential_Evolution(
                problema.fitness,
                ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                bounds_constraints=bounds[constraint_name],
                bounds=(problema.SUPERIOR, problema.INFERIOR),
                g_functions=problema.rest_g,
                h_functions=problema.rest_h,
                centroid=(constraint_name == "centroid"),
                beta=(constraint_name == "beta"),
                evolutionary=(constraint_name == "evolutionary"),
                res_and_rand=(constraint_name == "res_and_rand"),
            )
            algorithm.evolution(verbose=True)
            fitness_data.append(algorithm.gbest_fitness)
            violations_data.append(algorithm.gbest_violation)
            convergence_fitness_data.append(algorithm.gbest_fitness_list)
            convergence_violations_data.append(algorithm.gbest_violations_list)
        except Exception as e:
            print(f"Error en el metodo {constraint_name} en ejecución {_ + 1}: {e}")
        del algorithm
        gc.collect()

    factible_indices = [i for i, v in enumerate(violations_data) if v == 0]
    factible_fitness_data = [fitness_data[i] for i in factible_indices]

    # Guardar datos en archivos por restricción y problema
    save_results_to_csv_constraint(problem_name, constraint_name, fitness_data, violations_data)

    # Generar gráficos y tablas dependiendo de los resultados
    if len(factible_indices) == EXECUTIONS:
        plot_convergence_fitness(problem_name, constraint_name, convergence_fitness_data)
        plot_fitness_boxplot(problem_name, constraint_name, factible_fitness_data)
    else:
        save_violations_summary(problem_name, constraint_name, violations_data)

    # Guardar resultados en Excel con nombre aleatorio
    random_filename = generate_random_filename()

    return fitness_data, violations_data, convergence_violations_data, factible_fitness_data

def save_results_to_csv_constraint(problem_name, constraint_name, fitness_data, violations_data):
    directory = f'{DIRECTORY}/{constraint_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    df = pd.DataFrame({
        'Fitness': fitness_data,
        'Violations': violations_data
    })
    filename = f'{directory}/CEC2017_{problem_name}_{constraint_name}.csv'
    df.to_csv(filename, index=False)

def plot_convergence_fitness(problem_name, constraint_name, convergence_fitness_data):
    plt.figure(figsize=(10, 6))
    mean_convergence_fitness = [sum(gen) / len(gen) for gen in zip(*convergence_fitness_data)]
    plt.plot(mean_convergence_fitness, label=constraint_name)
    plt.title(f'Convergence Plot (Fitness) for {problem_name} with {constraint_name}')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    directory = f'{DIRECTORY}/{constraint_name}'
    filename = f'{directory}/{PROBLEM}_convergence_fitness_{problem_name}_{constraint_name}.png'
    plt.savefig(filename)
    plt.close()

def plot_fitness_boxplot(problem_name, constraint_name, factible_fitness_data):
    plt.figure(figsize=(12, 8))
    plt.boxplot(factible_fitness_data, labels=[constraint_name])
    plt.title(f'Box Plot de Fitness para {problem_name} con {constraint_name}')
    plt.xlabel('Restricción')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45)
    plt.grid(True)
    directory = f'{DIRECTORY}/{constraint_name}'
    filename = f'{directory}/{PROBLEM}_fitness_boxplot_{problem_name}_{constraint_name}.png'
    plt.savefig(filename)
    plt.close()

def save_violations_summary(problem_name, constraint_name, violations_data):
    """
    Guarda un resumen de las violaciones en un archivo CSV.

    Parámetros:
    problem_name (str): Nombre del problema.
    constraint_name (str): Nombre de la restricción.
    violations_data (list): Lista de datos de violaciones.
    """
    directory = f'{DIRECTORY}/{constraint_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    mean_violations = sum(violations_data) / len(violations_data) if violations_data else 0.0
    df = pd.DataFrame({
        'Constraint': [constraint_name],
        'Mean Violations': [mean_violations]
    })
    filename = f'{directory}/{PROBLEM}_violations_summary_{problem_name}.csv'
    df.to_csv(filename, index=False, mode='a' if os.path.exists(filename) else 'w')

def plot_convergence_violations_all(problem_name, all_convergence_data, bounds, log_scale=True, y_lim=None):
    """
    Genera una gráfica de convergencia de violaciones para todas las restricciones.

    Parámetros:
    problem_name (str): Nombre del problema.
    all_convergence_data (dict): Datos de convergencia de todas las restricciones.
    bounds (list): Lista de nombres de restricciones.
    log_scale (bool): Si True, usa escala logarítmica para el eje Y. Default es False.
    y_lim (tuple): Si se proporciona, establece los límites del eje Y (min, max).
    """
    plt.figure(figsize=(10, 6))
    for constraint_name in bounds:
        if constraint_name in all_convergence_data:
            mean_convergence_violations = [sum(gen) / len(gen) for gen in zip(*all_convergence_data[constraint_name])]
            plt.plot(mean_convergence_violations, label=constraint_name)
    
    plt.title(f'Convergence Plot (Violations) for {problem_name}')
    plt.xlabel('Generations')
    plt.ylabel('Violations')
    plt.legend()
    plt.grid(True)

    if log_scale:
        plt.yscale('log')

    if y_lim is not None:
        plt.ylim(y_lim)

    directory = DIRECTORY
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{directory}/{PROBLEM}_convergence_violations_{problem_name}.png'
    plt.savefig(filename)
    plt.close()


def plot_results_all(problem_name, all_violations_data, bounds):
    plt.figure(figsize=(12, 8))
    data = [violations for violations in all_violations_data.values()]
    labels = [constraint for constraint in all_violations_data.keys()]
    plt.boxplot(data, labels=labels)
    plt.title(f'Box Plot de Violations para {problem_name}')
    plt.xlabel('Restricción')
    plt.ylabel('Violations')
    plt.xticks(rotation=45)
    plt.grid(True)
    directory = 'report/cec2017'
    filename = f'{directory}/CEC2017_boxplot_violations_{problem_name}.png'
    plt.savefig(filename)
    plt.close()
    
def plot_fitness_boxplot_all(problem_name, all_fitness_data, bounds):
    plt.figure(figsize=(12, 8))
    data = [fitness for fitness in all_fitness_data.values()]
    labels = [constraint for constraint in all_fitness_data.keys()]
    plt.boxplot(data, labels=labels)
    plt.title(f'Box Plot de Fitness para {problem_name}')
    plt.xlabel('Restricción')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45)
    plt.grid(True)
    directory = DIRECTORY
    filename = f'{directory}/{PROBLEM}_fitness_boxplot_{problem_name}.png'
    plt.savefig(filename)
    plt.close()


def generate_random_filename():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))


def save_results_to_csv(results, filename):
    """
    Guarda los resultados en un archivo CSV.

    Parámetros:
    results (list): Lista de resultados a guardar.
    filename (str): Nombre del archivo CSV (sin extensión).
    """
    directory = DIRECTORY
    if not os.path.exists(directory):
        os.makedirs(directory)
    df = pd.DataFrame(results)
    filepath = f'{directory}/{filename}.csv'
    df.to_csv(filepath, index=False)
    

def plot_fitness_boxplot_from_csvs(directory, problem_name, exclude=[]):
    """
    Genera un boxplot de fitness excluyendo ciertas restricciones basándose en los archivos CSV generados.

    Parámetros:
    directory (str): Directorio donde se encuentran los archivos CSV.
    problem_name (str): Nombre del problema a analizar.
    exclude (list): Lista de restricciones a excluir del boxplot.
    """
    fitness_data = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and problem_name in file:
                constraint_name = root.split(os.sep)[-1]  # Obtener el nombre de la restricción
                if constraint_name not in exclude:
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    if 'Fitness' in df.columns:
                        fitness_data[constraint_name] = df['Fitness'].dropna().values

    if not fitness_data:
        print(f"No se encontraron datos de fitness para el problema {problem_name} con las restricciones especificadas.")
        return

    # Ordenar los datos por el valor medio del fitness
    sorted_fitness_data = sorted(fitness_data.items(), key=lambda x: x[1].mean())

    # Desempaquetar los datos ordenados
    sorted_labels, sorted_data = zip(*sorted_fitness_data)

    plt.figure(figsize=(12, 8))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.title(f'Box Plot de Fitness para {problem_name}')
    plt.xlabel('Restricción')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(f'{directory}/{PROBLEM}_fitness_boxplot_{problem_name}_{generate_random_filename()}.png')
    plt.close()

def plot_violations_boxplot_from_csvs(directory, problem_name, exclude=[]):
    """
    Genera un boxplot de violaciones excluyendo ciertas restricciones basándose en los archivos CSV generados.

    Parámetros:
    directory (str): Directorio donde se encuentran los archivos CSV.
    problem_name (str): Nombre del problema a analizar.
    exclude (list): Lista de restricciones a excluir del boxplot.
    """
    violations_data = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and problem_name in file:
                constraint_name = root.split(os.sep)[-1]  # Obtener el nombre de la restricción
                if constraint_name not in exclude:
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    if 'Violations' in df.columns:
                        violations_data[constraint_name] = df['Violations'].dropna().values

    if not violations_data:
        print(f"No se encontraron datos de violaciones para el problema {problem_name} con las restricciones especificadas.")
        return

    # Ordenar los datos por el valor medio de las violaciones
    sorted_violations_data = sorted(violations_data.items(), key=lambda x: x[1].mean())

    # Desempaquetar los datos ordenados
    sorted_labels, sorted_data = zip(*sorted_violations_data)

    plt.figure(figsize=(12, 8))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.title(f'Box Plot de Violations para {problem_name}')
    plt.xlabel('Restricción')
    plt.ylabel('Violations')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(f'{directory}/{PROBLEM}_violations_boxplot_{problem_name}_{generate_random_filename()}.png')
    plt.close()
    
