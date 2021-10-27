import pandas as pd
import os
from natsort import natsorted

#Preprocesamiento  de signos vitales con fecha
# names of interest
metrics_names = ["TENSION ARTERIAL MEDIA", "TENSIÓN ARTERIAL", "FRECUENCIA CARDIACA", "FRECUENCIA RESPIRATORIA",
                 "OXIMETRÍA", "TEMPERATURA"]

# LOOP #

# read every file name

os.chdir("..")
path = os.path.abspath(os.getcwd()) + '\\pacientes\\'
files = os.listdir(path)

# natural order for files string

files = natsorted(files)

# efficient storing of dataframes in list
small_dfs = []
small_dfs_2 = []

for file in files:
    df = pd.read_excel(path + file)

    # leave only the rows containing the desired metrics
    df = df[df['Unnamed: 4'].str.contains('|'.join(metrics_names), na=False)]

    # leave column with metric name and value
    columns_size = df.shape[1]

    if columns_size < 30:
        df = df[[df.columns[0], 'Unnamed: 4', 'Unnamed: 26']]
        val_col = 'Unnamed: 26'
    else:
        df = df[[df.columns[0], 'Unnamed: 4', 'Unnamed: 31']]
        val_col = 'Unnamed: 31'

    # change large names for abreviations
    df = df.replace({'Unnamed: 4': {'TENSION ARTERIAL MEDIA': 'TAM', 'TENSIÓN ARTERIAL': 'TA',
                                    'FRECUENCIA CARDIACA': 'FC', 'FRECUENCIA RESPIRATORIA': 'FR',
                                    'OXIMETRÍA': 'OXI', 'TEMPERATURA': 'TEMP'}})

    df = df.drop_duplicates(subset=[df.columns[0], 'Unnamed: 4'], keep='first')
    df = df.pivot(index=df.columns[0], columns='Unnamed: 4', values=val_col)

    #df = df.apply(lambda x: pd.Series(x.dropna().to_numpy()))

    if "-" in file:
        x=file.split("-",1)
        df['Paciente']=x[0]
    elif "_" in file:
        x=file.split("_",1)
        df['Paciente']=x[0]
    else:
        df['Paciente'] = file[:-4]

    small_dfs.append(df)
    print(file + ' done!')

final_df = pd.concat(small_dfs, ignore_index=False)
path = os.path.abspath(os.getcwd())
final_df.to_excel(path + "\\output\\signos_vitales_fecha_2.xlsx")
print(small_dfs)
