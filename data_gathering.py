# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:33:10 2021

@author: User
"""

import pandas as pd

# FOR ONE CASE #

# read data
df = pd.read_excel (r'C:\\Users\\User\\Desktop\\Modelo UniAndes\\pacientes\\163.xlsx')

# names of interest
metrics_names = ["TENSION ARTERIAL MEDIA","TENSIÓN ARTERIAL","FRECUENCIA CARDIACA","FRECUENCIA RESPIRATORIA","OXIMETRÍA","TEMPERATURA"]

# leave only the rows containing the desired metrics
df = df[df['Unnamed: 4'].str.contains('|'.join(metrics_names), na = False)]

# leave column with metric name and value
df = df[['Unnamed: 4','Unnamed: 31']]

# change large names for abreviations
df = df.replace({'Unnamed: 4' : { 'TENSION ARTERIAL MEDIA' : 'TAM', 'TENSIÓN ARTERIAL' : 'TA', 'FRECUENCIA CARDIACA' : 'FC', 'FRECUENCIA RESPIRATORIA': 'FR',
                            'OXIMETRÍA': 'OXI', 'TEMPERATURA': 'TEMP' }})

# pivot table
df = df.pivot(index=None, columns='Unnamed: 4', values='Unnamed: 31')

# remove nan and collapse rows
df = df.apply(lambda x: pd.Series(x.dropna().to_numpy()))

# add column of patient id
df['Paciente'] = 163

# order columns
df = df[['Paciente','FC', 'TA', 'TAM', 'TEMP', 'OXI', 'FR']]



#-----------------------------------------------------------------------------
import pandas as pd

# names of interest
metrics_names = ["TENSION ARTERIAL MEDIA","TENSIÓN ARTERIAL","FRECUENCIA CARDIACA","FRECUENCIA RESPIRATORIA","OXIMETRÍA","TEMPERATURA"]

# LOOP #

# read every file name
import os
files = os.listdir('C:\\Users\\User\\Desktop\\Modelo UniAndes\\pacientes\\')

# natural order for files string
from natsort import natsorted
files = natsorted(files)

# efficient storing of dataframes in list
small_dfs = []

for file in files:
    df = pd.read_excel (r'C:\\Users\\User\\Desktop\\Modelo UniAndes\\pacientes\\' + file)
    # leave only the rows containing the desired metrics
    df = df[df['Unnamed: 4'].str.contains('|'.join(metrics_names), na = False)]
    # leave column with metric name and value
    df = df[['Unnamed: 4','Unnamed: 31']]
    # change large names for abreviations
    df = df.replace({'Unnamed: 4' : { 'TENSION ARTERIAL MEDIA' : 'TAM', 'TENSIÓN ARTERIAL' : 'TA', 'FRECUENCIA CARDIACA' : 'FC', 'FRECUENCIA RESPIRATORIA': 'FR',
                            'OXIMETRÍA': 'OXI', 'TEMPERATURA': 'TEMP' }})
    # pivot table
    df = df.pivot(index=None, columns='Unnamed: 4', values='Unnamed: 31')
    # remove nan and collapse rows
    df = df.apply(lambda x: pd.Series(x.dropna().to_numpy()))
    # add column of patient id (remove .xlsx suffix)
    df['Paciente'] = file[:-4]
    # order columns
    df = df[['Paciente','FC', 'TA', 'TAM', 'TEMP', 'OXI', 'FR']]
    # append every df to the list
    small_dfs.append(df)
    
    print(file + ' done!')
    
# build final dataframe
final_df = pd.concat(small_dfs, ignore_index=True)
    

# export data
final_df.to_excel("C:\\Users\\User\\Desktop\\Modelo UniAndes\\output\\signos_vitales.xlsx")








    
    
    
    
    
    
    
    
    
    
    
    
    