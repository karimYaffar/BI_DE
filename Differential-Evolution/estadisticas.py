import pandas as pd
from scipy.stats import wilcoxon

def compare_proposals_from_excel(file_path_proposal, file_path_baseline, sheet_name, column_name):
    """
    Compara dos listas de valores en dos archivos Excel diferentes utilizando la prueba de Wilcoxon.
    
    Parameters:
    file_path_proposal (str): Ruta al archivo Excel de la propuesta.
    file_path_baseline (str): Ruta al archivo Excel del estándar de comparación.
    sheet_name (str): Nombre de la hoja de Excel (asumiendo que es la misma en ambos archivos).
    column_name (str): Nombre de la columna de "Fitness" en ambos archivos.
    
    Returns:
    str: Resultado de la comparación, incluyendo los arrays, medias y valor p.
    """

    
    # Leer los archivos Excel
    df_proposal = pd.read_excel(file_path_proposal, sheet_name=sheet_name)
    df_baseline = pd.read_excel(file_path_baseline, sheet_name=sheet_name)
    
    # Extraer los datos de la columna especificada
    proposal = df_proposal[column_name].dropna().values
    baseline = df_baseline[column_name].dropna().values
    
    if len(proposal) != len(baseline):
        raise ValueError("Las listas deben tener la misma longitud.")
    
    # Calcular las medias
    mean_proposal = proposal.mean()
    mean_baseline = baseline.mean()
    
    # Aplicar la prueba de Wilcoxon unilateral (menor es mejor)
    stat, p_value = wilcoxon(baseline, proposal, alternative='less')
    
    # Determinar el resultado
    alpha = 0.05
    if p_value < alpha:
        result = "La diferenci es estadisticamente significativa."
        if mean_proposal<mean_baseline:
            result = result + "(gana mean_proposal)."
        else:
            result = result + "(gana mean_baseline)."
            
    else:
        result = "No hay una diferencia estadísticamente significativa (expate) "
    
    
    # Imprimir arrays, medias, valor p y resultado
    output = ( 
              f"Problema:C28 \n"
              f"Media de la propuesta(D): {mean_proposal}\n"
              f"Media de la línea base: {mean_baseline}\n"
              f"Valor p: {p_value}\n"
              f"Valor stat: {stat}\n"
              f"Resultado: {result}")
    
    return output

# Ejemplo de uso con archivos Excel

DIRECTORY = "CEC2017/C01"
CURRENT_PROBLEM = "CEC2017_C01_"
BOUNDARY_PROPOSAL = "Reflection_"
ALGORTIM_PROPOSAL = "DE-rand-1-bin_"

DIRECTORY_PROPOSAL = f"{DIRECTORY}/{CURRENT_PROBLEM}{BOUNDARY_PROPOSAL}{ALGORTIM_PROPOSAL}.xlsx"

BOUNDARY_BASELINE = "Centroid_"
ALGORTIM_BASELINE = "DMDE-AP_"
DIRECTORY_BASELINE = f"{DIRECTORY}/{CURRENT_PROBLEM}{BOUNDARY_BASELINE}{ALGORTIM_BASELINE}.xlsx"



file_path_proposal =   DIRECTORY_PROPOSAL # Ruta al archivo de la propuesta
file_path_baseline =   DIRECTORY_BASELINE # Ruta al archivo de comparación
sheet_name = 'Sheet1'  # Nombre de la hoja de Excel
column_name = 'Fitness'  # Nombre de la columna de Fitness

result = compare_proposals_from_excel(file_path_proposal, file_path_baseline, sheet_name, column_name)
print(result)




