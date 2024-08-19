import pandas as pd
import matplotlib.pyplot as plt

def create_box_plot_for_xlsx(filename: str, savefilename: str):
    # Load the Excel file
    xls = pd.ExcelFile(filename)
    
    # For each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Ensure the dataframe has the necessary columns
        if {"Problema", "Restricción", "Fitness Promedio", "Ejecuciones Factibles"}.issubset(df.columns):
            problems = df["Problema"].unique()
            
            for problem in problems:
                problem_data = df[df["Problema"] == problem]
                
                # Create the box plot
                plt.figure(figsize=(12, 8))
                data = [problem_data[problem_data["Restricción"] == constraint]["Fitness Promedio"] for constraint in problem_data["Restricción"].unique()]
                labels = problem_data["Restricción"].unique()
                plt.boxplot(data, labels=labels)
                plt.title(f'Box Plot de Fitness Promedio para {problem}')
                plt.xlabel('Restricción')
                plt.ylabel('Fitness Promedio')
                plt.xticks(rotation=45)
                plt.grid(True)
                
                # Save the plot
                plt.savefig(f'{savefilename}_{problem}.png')
                plt.close()
