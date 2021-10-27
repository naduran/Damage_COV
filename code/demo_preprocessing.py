# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:42:10 2021

@author: Useron
"""
# ---------------------------#
# DEMOGRÁFICOS DATA SET
# ---------------------------#

# import libraries

# pip install panda
# pip install numpy
import pandas as pd
import numpy as np
import os

os.chdir("..")
path = os.path.abspath(os.getcwd())


# Read data
data = pd.read_excel(path + '\\input\\base_demo.xlsx')

# Identify anomalies
data.describe()

# number of rows
data.shape[0]


# ------------------------------------------------------------------------------
# SELECT DATA

# using np.r_ for multi-selection
# s
df = data.iloc[np.r_[0:data.shape[0]], :]
#df = df.fillna(0)
# drop rows with mostly nas
# df = df.drop([df.index[135]])

# remove white spaces at both ends for column names
df.columns = df.columns.str.strip()

# remove useless columns (sintomas and comorbilidad name columns)
# df = df.drop(df.columns[[3, 22,56,57,58,59,60]], axis=1) #Ajuste con datos de interes
df = df.drop(df.columns[[3, 22]], axis=1)
# remove unnamed empty columns
# df = df.iloc[:, 0:56] #Ajuste con datos de interes
df = df.iloc[:, 0:54]


# replace nas with zeroes / considering the random forest takes care of nas lets leave it like this for now
df = df.replace(np.nan, 0)


# dealing with 'otro' column: for the moment leave it as 1 and 0


def change_other(x):
    if x == "No" or x == "No ":
        return 0
    else:
        return 1


# rename column 'otro'
# df = df.rename(columns={df.columns[20]: 'otro sintoma'})

# apply function

# df['otro sintoma'] = df.apply(change_other, axis=1)

df['otro sintoma'] = df['otro sintoma'].apply(change_other)

# df[df.columns[20]] = df.apply(lambda x: change_other, axis=1)




# comorbilidad
# dealing with 'otro' column


def change_other2(row):
    if row == "No" or row == "No ":
        return 0
    else:
        return 1


# rename column 'otro'
df = df.rename(columns={df.columns[51]: 'otra comorbilidad'})

# apply function
# df['otra comorbilidad'] = df.apply(change_other2, axis=1)
# df[df.columns[51]] = df.apply(lambda x: change_other2, axis=1)
df[df.columns[51]] = df[df.columns[51]].apply(change_other)

# remove white spaces at both ends
df.columns = df.columns.str.strip()

# remove white spaces from Genero column
df['Genero'] = df['Genero'].str.strip()

# Genero column encoding
cleanup_col = {"Genero": {"Masculino": 0, "Femenino": 1},
               "Ventilacion": {1: "Invasiva", 0: "No invasiva"}}

df = df.replace(cleanup_col)

# encoding of Edad column
df.loc[df['Edad'] < 70, 'Edad'] = 5
df.loc[df['Edad'] >= 70, 'Edad'] = 0


# remove white spaces at both ends
df.columns = df.columns.str.strip()
# df.rows = df.rows.str.strip()


# VERY IMPORTANT: if the next values are present in the data the h2o model won't train
# replace number in response feature
df.loc[df['Ventilacion'] == 6, 'Ventilacion'] = "Invasiva"

# remove na in response variable
df = df[df['Ventilacion'].notna()]

# ------------------------------------------------------------------------------
# CATEGORICAL VARIABLES ENCODING

df = df.rename(columns={"Dolor toraxico": "Dolor toracico", "Dolor abdolimal": "Dolor abdominal",
                        "Pedida del apetito": "Perdida del apetito",
                        "Otra enfermedad carfiaca": "Otra enfermedad cardiaca"})

# replace values by weights (síntomas)
df.loc[(df['Tos'] == 1), 'Tos'] = 5
df.loc[(df['Diarrea'] == 1), 'Diarrea'] = 3
df.loc[(df[
            'Dificultad respiratoria si disnea y si taquipnea'] == 1), 'Dificultad respiratoria si disnea y si taquipnea'] = 5
df.loc[(df['Dolor abdominal'] == 1), 'Dolor abdominal'] = 1
df.loc[(df['Dolor toracico'] == 1), 'Dolor toracico'] = 4
df.loc[(df['Escalofrios'] == 1), 'Escalofrios'] = 4
df.loc[(df['Fiebre'] == 1), 'Fiebre'] = 5
df.loc[(df['Malestar General'] == 1), 'Malestar General'] = 4
df.loc[(df['Mialgia'] == 1), 'Mialgia'] = 3
df.loc[(df['Nauseas'] == 1), 'Nauseas'] = 3
df.loc[(df['Odinofagia'] == 1), 'Odinofagia'] = 3
df.loc[(df['otro sintoma'] == 1), 'otro sintoma'] = 1
df.loc[(df['Hiporexia (Pedida del apetito )'] == 1), 'Hiporexia (Pedida del apetito )'] = 1
df.loc[(df['Anosmia(Perdida del olfato )'] == 1), 'Anosmia(Perdida del olfato )'] = 5
df.loc[(df['Cefalea'] == 1), 'Cefalea'] = 3
df.loc[(df['Taquipnea'] == 1), 'Taquipnea'] = 5
# df.loc[(df['Vomito'] == 1),'Vomito'] = 3


# replace values by weights (comorbilidades)
df.loc[(df['Asma'] == 1), 'Asma'] = 3
df.loc[(df['EPOC'] == 1), 'EPOC'] = 5
df.loc[(df['Diabetes'] == 1), 'Diabetes'] = 5
df.loc[(df['VIH'] == 1), 'VIH'] = 2
df.loc[(df['Enfermedad coronaria'] == 1), 'Enfermedad coronaria'] = 5
df.loc[(df['Falla Cardiaca'] == 1), 'Falla Cardiaca'] = 5
df.loc[(df['Enfermedad Valvular'] == 1), 'Enfermedad Valvular'] = 5
df.loc[(df['Otra enfermedad cardiaca'] == 1), 'Otra enfermedad cardiaca'] = 5
df.loc[(df['Cancer'] == 1), 'Cancer'] = 5
df.loc[(df['Desnutricion'] == 1), 'Desnutricion'] = 3
df.loc[(df['Obesidad'] == 1), 'Obesidad'] = 3
df.loc[(df['Enfermedad renal'] == 1), 'Enfermedad renal'] = 3
# df.loc[(df['Toma medicamentos inmunosupresores'] == 1),'Toma medicamentos inmunosupresores'] = 3
df.loc[(df['Tabaquismo'] == 1), 'Tabaquismo'] = 4
df.loc[(df['Tuberculosis'] == 1), 'Tuberculosis'] = 2
df.loc[(df['Hipertension'] == 1), 'Hipertension'] = 5
df.loc[(df['Enfermedades reumaticas'] == 1), 'Enfermedades reumaticas'] = 3
df.loc[(df['Transtornos neurologicos cronicos'] == 1), 'Transtornos neurologicos cronicos'] = 2
df.loc[(df['Enfermedad hematologica cronica'] == 1), 'Enfermedad hematologica cronica'] = 2
df.loc[(df['Enfermedad hepatica cronica'] == 1), 'Enfermedad hepatica cronica'] = 2
df.loc[(df['Alcoholismo'] == 1), 'Alcoholismo'] = 2
df.loc[(df['otra comorbilidad'] == 1), 'otra comorbilidad'] = 1
df.loc[(df['Ninguno'] == 1), 'Ninguno'] = 1
df.loc[(df['Artritis reumatoide'] == 1), 'Artritis reumatoide'] = 1
df.loc[(df['Psoriasis'] == 1), 'Psoriasis'] = 1

# to add more...

# ------------------------------------------------------------------------------


# # select every column except for "Paciente"
# cols=[i for i in df.columns if i not in ["Paciente"]]
# #cols=[i for i in df.columns]

# for col in cols:
#     df[col] = df[col].astype('category')

# save clean df
df.to_excel(path + "\\output\\demo_clean.xlsx", sheet_name='sheet1')

# ------------------------------------------------------------------------------
