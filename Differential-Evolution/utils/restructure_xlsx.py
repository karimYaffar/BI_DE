import pandas as pd


def restructure_xlsx_to_new_format(filename: str, savefilename: str):
    # Load the Excel file
    xls = pd.ExcelFile(filename)
    
    # Collect all the necessary data
    data = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Ensure the dataframe has the necessary columns
        if {"Problema", "Restricción", "Fitness Promedio", "Ejecuciones Factibles"}.issubset(df.columns):
            problems = df["Problema"].unique()
            
            for problem in problems:
                problem_data = df[df["Problema"] == problem]
                row = {"Problema": problem}
                
                for _, row_data in problem_data.iterrows():
                    restriction = row_data["Restricción"]
                    fitness_promedio = row_data["Fitness Promedio"]
                    ejecuciones_factibles = row_data["Ejecuciones Factibles"]
                    row[f'{restriction}'] = fitness_promedio
                    row[f'{restriction}_EF'] = ejecuciones_factibles
                
                data.append(row)

    # Convert the collected data to a DataFrame
    result_df = pd.DataFrame(data)

    # Rename columns to have consistent "EF" naming
    cols = list(result_df.columns)
    for i in range(2, len(cols), 2):
        cols[i] = "EF"
    result_df.columns = cols

    # Save the DataFrame to a new Excel file
    result_df.to_excel(savefilename, index=False)

# Run the function for the provided file
restructure_xlsx_to_new_format('../C01-C03 centroid.xlsx', './restructured_C01-C03_new_format.xlsx')
