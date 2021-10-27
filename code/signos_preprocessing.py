# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 00:38:50 2021

@author: User
"""

import pandas as pd
import numpy as np
import os

os.chdir("..")
path = os.path.abspath(os.getcwd())
print(path)


df = pd.read_excel(path + '\\output\\demo_clean.xlsx')

# Read data
sv = pd.read_excel(path + '\\output\\signos_vitales.xlsx')
print("Primer filtro")
print(sv)
sv = sv.drop(sv.columns[[0]], axis = 1)

# remove white spaces at both ends
#sv.columns = sv.columns.str.strip()

sv['Paciente'] = sv['Paciente'].str.replace('Paciente ', '')
#print(sv)
dfsub = df.iloc[:,np.r_[1:2,54:55],]

# dfsub = df.iloc[:,np.r_[1:2,54:55,55:56,56:57],] #Cambio variables de interes
#print(dfsub)

sv['Paciente']=sv['Paciente'].astype(int)

# join dataframes
sv2 = sv.merge(dfsub, on='Paciente', how='left')
print(sv2)

sv2.dtypes
dfsub.dtypes

# drop Paciente and TM columns
sv2 = sv2.drop(sv2.columns[[3]], axis = 1)
print(sv2)

sv2.dtypes

# separate TA column
sv2[['TA SISTOLICA', 'TA DIASTOLICA']] = sv2['TA'].str.split('/', 1, expand=True)
sv2 = sv2.drop(sv2.columns[[2]], axis = 1)


# convert TEMP to int
sv2["TEMP"] = sv2["TEMP"].str.replace(',', '.')

sv2["TEMP"] = pd.to_numeric(sv2["TEMP"])

#Esto se usa para borrar vacíos
sv2 = sv2.dropna()

sv2["TA SISTOLICA"] = pd.to_numeric(sv2["TA SISTOLICA"])
sv2["TA DIASTOLICA"] = pd.to_numeric(sv2["TA DIASTOLICA"])

# create column based on formula
# TAM = (S+2*D)/3

sv2["TAM"] = (sv2["TA SISTOLICA"] + (2 * sv2["TA DIASTOLICA"]) ) / 3

sv2 = sv2[['FC', 'TA SISTOLICA', 'TA DIASTOLICA', 'TAM', 'TEMP', 'OXI', 'FR', 'Ventilacion']]
# sv2 = sv2[['Paciente','FC', 'TA SISTOLICA', 'TA DIASTOLICA', 'TAM', 'TEMP', 'OXI', 'FR','Ventilacion','Fecha de inicio de maxima ventilacion', 'Mortalidad']]


# save clean df
sv2.to_excel(path + "\\output\\signos_vitales_clean.xlsx", sheet_name='sheet1')
