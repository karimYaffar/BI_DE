# Limites y restricciones
from core.bchms import BCHM
from core.constraints_functions import ConstriantsFunctionsHandler
import json 

# Estrategias de velocidad
from utils.constants import EXECUTIONS, SIZE_POPULATION

# Funciones objetivas
from functions.cec2006problems import *
from functions.cec2020problems import *
from functions.cec2022problems import *
from functions.cec2017problems import *
from functions.cec2010problems import *
import os 

from core.differential_evolution import Differential_Evolution


from core.BI_DE import BI_DE


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parámetros del problema y estrategias
problema = CEC2017_C05()
strategies = ['DE-rand-1-bin'  ]
# 'DE-rand-1-bin' , 'DE-CTB-1-bin'
def main():
    problem = "CEC2010_C05"
    
   
    
    for strategy in strategies:
        fitness_data_de = []
        violations_data_de = []
        first_gen = []
        

        
           
          
        for i in range(1, EXECUTIONS + 1):
            print(f"Ejecución: {i} con estrategia: {strategy}")
            
            de = Differential_Evolution(
                problema.fitness,
                ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                BCHM.reflection,
                (problema.SUPERIOR, problema.INFERIOR),
                problema.rest_g,
                problema.rest_h,
                strategy=strategy,
                centroid=False,
                # mirror_reflect
            )
            limite = "Reflection"
            de.evolution()
            fitness_data_de.append(de.best_fitness)
            violations_data_de.append(de.best_violations)
            first_gen.append(de.first_feasible_generation)
        save_results_to_excel(problem, limite, strategy, fitness_data_de, violations_data_de,first_gen ,'')
        
        
        
        ########################################
        """ 
        for i in range(1, EXECUTIONS + 1):
            print(f"Ejecución: {i} con algortimo: APDE_NEW4")
            
            de = BI_DE(
                problema.fitness,
                ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                BCHM.centroid,
                (problema.SUPERIOR, problema.INFERIOR),
                problema.rest_g,
                problema.rest_h,
                centroid=True,
            )
            de.evolution()
            fitness_data_de.append(de.best_fitness)
            violations_data_de.append(de.best_violations)
            first_gen.append(de.first_feasible_generation)
           
        limite = "centroid"
        save_results_to_excel(problem, limite, 'DMDE-AP', fitness_data_de, violations_data_de, first_gen,  '')
         """
        ########################################

        

         
           
       
        
       
def save_results_to_excel(problema, restriccion, estrategia, fitness_data, violations_data, first_gen ,algorithm_name):
    # Preparar los datos para el DataFrame
    rows = []
    for fitness, violations, first_gen  in zip(fitness_data, violations_data, first_gen):
        rows.append([fitness, violations, first_gen])
    
    # Crear el DataFrame
    df = pd.DataFrame(rows, columns=["Fitness", "Violaciones", "generacion_factible" ])
    
    # Nombre del archivo basado en problema, restricción, estrategia y algoritmo
    filename = f"{problema}_{restriccion}_{estrategia}_{algorithm_name}.xlsx"
    
    # Guardar el DataFrame en un archivo Excel
    df.to_excel(filename, index=False)
    print(f"Resultados guardados en '{filename}'")

    plt.show()


def plot_fitness_boxplot_from_excels(directory, problem_name, exclude=[]):
    """
    Genera un boxplot de fitness excluyendo ciertas restricciones basándose en los archivos Excel generados.

    Parámetros:
    directory (str): Directorio donde se encuentran los archivos Excel.
    problem_name (str): Nombre del problema a analizar.
    exclude (list): Lista de restricciones a excluir del boxplot.
    """
    fitness_data = {}
    Violaciones = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx") and problem_name in file:
                # Obtener el nombre de la restricción y el porcentaje combinado
                parts = file.split('_')
                constraint_name = f"{parts[3]}"  # Obtener el nombre de la restricción y el porcentaje combinado
                if constraint_name not in exclude:
                    file_path = os.path.join(root, file)
                    df = pd.read_excel(file_path)
                    if 'Fitness' in df.columns:
                        if constraint_name in fitness_data:
                            fitness_data[constraint_name].extend(df['Fitness'].dropna().values)
                        else:
                            fitness_data[constraint_name] = df['Fitness'].dropna().values.tolist()

    if not fitness_data:
        print(f"No se encontraron datos de fitness para el problema {problem_name} con las restricciones especificadas.")
        return

    # Ordenar los datos por el valor medio del fitness
    sorted_fitness_data = sorted(fitness_data.items(), key=lambda x: pd.Series(x[1]).mean())

    # Desempaquetar los datos ordenados
    sorted_labels, sorted_data = zip(*sorted_fitness_data)

    plt.figure(figsize=(12, 8))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.title('')
    #plt.title(f'Box Plot de Fitness para {problem_name}')
    plt.xlabel('Algortimos')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45)
    plt.grid(True)
    #plt.yscale('log')
    plt.savefig(f'{directory}/CEC2017_fitness_boxplot_{problem_name}_2.png')
    plt.close()

###########################################
def plot_violations_boxplot_from_excels(directory, problem_name, exclude=[]):
    """
    Genera un boxplot de violaciones excluyendo ciertas restricciones basándose en los archivos Excel generados.

    Parámetros:
    directory (str): Directorio donde se encuentran los archivos Excel.
    problem_name (str): Nombre del problema a analizar.
    exclude (list): Lista de restricciones a excluir del boxplot.
    """
    violations_data = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx") and problem_name in file:
                # Obtener el nombre de la restricción y el porcentaje combinado
                parts = file.split('_')
                constraint_name = f"{parts[3]}"  # Obtener el nombre de la restricción y el porcentaje combinado
                if constraint_name not in exclude:
                    file_path = os.path.join(root, file)
                    df = pd.read_excel(file_path)
                    if 'Violaciones' in df.columns or 'Violations' in df.columns:
                        column_name = 'Violaciones' if 'Violaciones' in df.columns else 'Violations'
    
                        if constraint_name in violations_data:
                            violations_data[constraint_name].extend(df[column_name].dropna().values)
                        else:
                            violations_data[constraint_name] = df[column_name].dropna().values.tolist()

    if not violations_data:
        print(f"No se encontraron datos de violaciones para el problema {problem_name} con las restricciones especificadas.")
        return

    # Ordenar los datos por el valor medio de las violaciones
    sorted_violations_data = sorted(violations_data.items(), key=lambda x: pd.Series(x[1]).mean())

    # Desempaquetar los datos ordenados
    sorted_labels, sorted_data = zip(*sorted_violations_data)

    plt.figure(figsize=(12, 8))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.title(f'Box Plot de Violaciones para {problem_name}')
    plt.xlabel('Restricción')
    plt.ylabel('Violaciones')
    plt.xticks(rotation=45)
    plt.grid(True)
    #plt.yscale('log')
    plt.savefig(f'{directory}/CEC2017_violations_boxplot_{problem_name}_1.png')
    plt.close()
###########################################

if __name__ == "__main__":
    #main()
    plot_fitness_boxplot_from_excels('CEC2024/C04/', 'CEC2024_C04', exclude=[ 
    #'DE-CTB-1-bin'
    #, 'DE-rand-1-bin'
    #,'DMDE-AP'
    #'DEEP'
    ])
    

    #plot_violations_boxplot_from_excels('CEC2017/C23/', 'CEC2017_C23', exclude=[
    #'DE-CTB-1-bin'
    #'DE-rand-1-bin'
    #,'DMDE-AP'
    #'DEEP'
    #])
