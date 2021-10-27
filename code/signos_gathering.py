# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:33:10 2021

@author: User
"""

#Este algoritmo se utiliza para procesar los signos vitales de los reportes de enfermería de cada paciente

import pandas as pd

#-----------------------------------------------------------------------------
import pandas as pd
import os
from natsort import natsorted

# names of interest
metrics_names = ["TENSION ARTERIAL MEDIA", "TENSIÓN ARTERIAL", "FRECUENCIA CARDIACA", "FRECUENCIA RESPIRATORIA", "OXIMETRÍA", "TEMPERATURA"]


# LOOP #

# read every file name

os.chdir("..")
path = os.path.abspath(os.getcwd())+'\\pacientes\\'
files = os.listdir(path)

# natural order for files string

files = natsorted(files)

# efficient storing of dataframes in list
small_dfs = []
small_dfs_2 = []


for file in files:
    df = pd.read_excel(path + file)

    # leave only the rows containing the desired metrics
    df = df[df['Unnamed: 4'].str.contains('|'.join(metrics_names), na = False)]

    # leave column with metric name and value
    columns_size = df.shape[1]

    if columns_size < 30:
        df = df[['Unnamed: 4', 'Unnamed: 26', df.columns[0]]]
        val_col = 'Unnamed: 26'
    else:
        df = df[['Unnamed: 4', 'Unnamed: 31', df.columns[0]]]
        val_col = 'Unnamed: 31'

    # change large names for abreviations
    df = df.replace({'Unnamed: 4' : { 'TENSION ARTERIAL MEDIA' : 'TAM', 'TENSIÓN ARTERIAL' : 'TA', 'FRECUENCIA CARDIACA' : 'FC', 'FRECUENCIA RESPIRATORIA': 'FR',
                            'OXIMETRÍA': 'OXI', 'TEMPERATURA': 'TEMP' }})

    # pivot table
    df = df.pivot(index=None, columns='Unnamed: 4', values=val_col)

    # remove nan and collapse rows
    df = df.apply(lambda x: pd.Series(x.dropna().to_numpy()))

    # add column of patient id (remove .xlsx suffix)
    if "-" in file:
        x=file.split("-",1)
        df['Paciente']=x[0]
    elif "_" in file:
        x=file.split("_",1)
        df['Paciente']=x[0]
    else:
        df['Paciente'] = file[:-4]

    # order columns
    df = df[['Paciente','FC', 'TA', 'TAM', 'TEMP', 'OXI', 'FR']]

    # append every df to the list
    small_dfs.append(df)

    print(file + ' done!')
    
# build final dataframe
final_df = pd.concat(small_dfs, ignore_index=True)
    
path = os.path.abspath(os.getcwd())
# export data

final_df.to_excel(path+"\\output\\signos_vitales.xlsx")



    
    
    
    
    
    
    
    
    
    
    
    