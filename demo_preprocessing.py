# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:42:10 2021

@author: User
"""

import pandas as pd
import numpy as np

path = "C:\\Users\\User\\Desktop\\Modelo UniAndes"

# Read data
data = pd.read_excel(path + '\\input\\base_demo.xlsx')

data.head(5)
data.shape

# Identify anomalies
data.describe()



# number of rows
data.shape[0]
len(data.index)

#------------------------------------------------------------------------------

# edad, sexo, comorbilidad, síntomas (oximetría), uci o no uci

# seleccionar columnas deseadas
# demográficas (1, 2: edad y sexo )

# síntomas (4, 20)
# comorbilidades (22, 44)
# estancia en UCI 93
# fallecido o no 94

# select until 198 row index (update according to most recent data)
df = data.iloc[np.r_[0:72],:]

#data.iloc[:,[1,2]]


# using np.r_ for multi-selection
#df = data.iloc[:, np.r_[1:3, 4:21, 22:45, 93:95]]


# delete rows index 38 that have missing data
#patient | index
# 65:38
#df = df.drop(df.index[[38]])

# remove white spaces at both ends
df.columns = df.columns.str.strip()

# remove rows with nan in 'Edad'
#df = df[pd.notnull(df['Edad'])]


# remove Sintomas Comorbilidades oxigeno and scores columns
df = df.drop(df.columns[[3,22]], axis = 1)

# remover hasta la 53
df = df.iloc[:, 0:54]

# replace nas with zeroes
#df = df.replace(np.nan, 0)

# sintoma
# dealing with 'otro' column: for the moment leave it as 1 and 0
def change_other (row):
   if row['otro sintoma'] == "No" or row['otro sintoma'] == "No ":
      return 0
   else:
      return 1

# rename column 'otro'
df = df.rename(columns={df.columns[20]: 'otro sintoma'})

# apply function
df['otro sintoma'] = df.apply(change_other, axis=1)

# comorbildiad
# dealing with 'otro' column
def change_other2 (row):
   if row['otra comorbilidad'] == "No" or row['otra comorbilidad'] == "No ":
      return 0
   else:
      return 1

# rename column
#df = df.rename(columns={df.columns[50]: 'otra comorbilidad'})

# apply function
df['otra comorbilidad'] = df.apply(change_other2, axis=1)

# remove white spaces at both ends
df.columns = df.columns.str.strip()

# remove white spaces from Genero column
df['Genero'] = df['Genero'].str.strip()



# Genero column encoding
cleanup_col = {"Genero":     {"Masculino": 0, "Femenino": 1},
               "Ventilacion":{1:"Invasiva", 0: "No invasiva"}}

df = df.replace(cleanup_col)

#cleanup_col = {"Genero":     {0: "Masculino", 1: "Femenino"}}

#df = df.replace(cleanup_col)

# encoding of Edad column
df.loc[df['Edad'] < 70, 'Edad'] = 5
df.loc[df['Edad'] >= 70, 'Edad'] = 0

#------------------------------------------------------------------------------

# CATEGORICAL VARIABLES LABEL ENCODING

# since we are going to use a decision-tree based algorithm
# our ordinal categorical variables should use ordinal/label encoding
# assuming equal magnitude between values

# read new data
#data2 = pd.read_excel(path + '\\base_recortada.xlsx', sheet_name = 'puntuación sintomas y anteceden')

# remove every collumn full with nan
#data = data.dropna(axis=1, how='all')
#data = data.dropna(axis=0, how='all')

# remove single row with possible error
#data2.drop(21,axis=0,inplace=True)

# remove prefix
# def drop_prefix(self, prefix):
#     self.columns = self.columns.str.lstrip(prefix)
#     return self

#pd.core.frame.DataFrame.drop_prefix = drop_prefix

#df = df.drop_prefix('Síntomas/_')

# remove numbers and symbols from column names
#df.columns = df.columns.str.replace('[=,<]', '')
#df.columns = df.columns.str.replace('\d+', '')

# remove accents
#import unidecode
#df.columns = df.columns.str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8')

# remove white spaces at both ends
df.columns = df.columns.str.strip()


#------------------------------------------------------------------------------

df = df.rename(columns={"Dolor toraxico": "Dolor toracico", "Dolor abdolimal": "Dolor abdominal", "Pedida del apetito": "Perdida del apetito","Otra enfermedad carfiaca":"Otra enfermedad cardiaca"})


# replace values by weights (síntomas)
df.loc[(df['Tos'] == 1),'Tos'] = 5
df.loc[(df['Diarrea'] == 1),'Diarrea'] = 3
df.loc[(df['Dificultad respiratoria si disnea y si taquipnea'] == 1),'Dificultad respiratoria si disnea y si taquipnea'] = 5
df.loc[(df['Dolor abdominal'] == 1),'Dolor abdominal'] = 1
df.loc[(df['Dolor toracico'] == 1),'Dolor toracico'] = 4
df.loc[(df['Escalofrios'] == 1),'Escalofrios'] = 4
df.loc[(df['Fiebre'] == 1),'Fiebre'] = 5
df.loc[(df['Malestar General'] == 1),'Malestar General'] = 4
df.loc[(df['Mialgia'] == 1),'Mialgia'] = 3
df.loc[(df['Nauseas'] == 1),'Nauseas'] = 3
df.loc[(df['Odinofagia'] == 1),'Odinofagia'] = 3
df.loc[(df['otro sintoma'] == 1),'otro sintoma'] = 1
df.loc[(df['Perdida del apetito'] == 1),'Perdida del apetito'] = 1
df.loc[(df['Perdida del olfato'] == 1),'Perdida del olfato'] = 5
df.loc[(df['Cefalea'] == 1),'Cefalea'] = 3
df.loc[(df['Taquipenia'] == 1),'Taquipenia'] = 5
df.loc[(df['Vomito'] == 1),'Vomito'] = 3


# replace values by weights (comorbilidades)
df.loc[(df['Asma'] == 1),'Asma'] = 3
df.loc[(df['EPOC'] == 1),'EPOC'] = 5
df.loc[(df['Diabetes'] == 1),'Diabetes'] = 5
df.loc[(df['VIH'] == 1),'VIH'] = 2
df.loc[(df['Enfermedad coronaria'] == 1),'Enfermedad coronaria'] = 5
df.loc[(df['Falla Cardiaca'] == 1),'Falla Cardiaca'] = 5
df.loc[(df['Enfermedad Valvular'] == 1),'Enfermedad Valvular'] = 5
df.loc[(df['Otra enfermedad cardiaca'] == 1),'Otra enfermedad cardiaca'] = 5
df.loc[(df['Cancer'] == 1),'Cancer'] = 5
df.loc[(df['Desnutricion'] == 1),'Desnutricion'] = 3
df.loc[(df['Obesidad'] == 1),'Obesidad'] = 3
df.loc[(df['Enfermedad renal'] == 1),'Enfermedad renal'] = 3
#df.loc[(df['Toma medicamentos inmunosupresores'] == 1),'Toma medicamentos inmunosupresores'] = 3
df.loc[(df['Tabaquismo'] == 1),'Tabaquismo'] = 4
df.loc[(df['Tuberculosis'] == 1),'Tuberculosis'] = 2
df.loc[(df['Hipertension'] == 1),'Hipertension'] = 5
df.loc[(df['Enfermedades reumaticas'] == 1),'Enfermedades reumaticas'] = 3
df.loc[(df['Transtornos neurologicos cronicos'] == 1),'Transtornos neurologicos cronicos'] = 2
df.loc[(df['Enfermedad hematologica cronica'] == 1),'Enfermedad hematologica cronica'] = 2
df.loc[(df['Enfermedad hepatica cronica'] == 1),'Enfermedad hepatica cronica'] = 2
df.loc[(df['Alcoholismo'] == 1),'Alcoholismo'] = 2
df.loc[(df['otra comorbilidad'] == 1),'otra comorbilidad'] = 1
df.loc[(df['Ninguno'] == 1),'Ninguno'] = 1
df.loc[(df['Artritis reumatoide'] == 1),'Artritis reumatoide'] = 1
df.loc[(df['Psoriasis'] == 1),'Psoriasis'] = 1

# to add more...

# save clean df
df.to_excel(path + "\\output\\demo_clean.xlsx", sheet_name='sheet1')

#------------------------------------------------------------------------------

# fix column name
df = df.rename(columns={'exo': 'Sexo'})
df = df.rename(columns={'Otros': 'Otro sintoma'})
df = df.rename(columns={'Ninguno': 'Ninguna comorbilidad'})

df["Estancia en UCI"].value_counts()
df["Sexo"].value_counts()
df["Estado del paciente a la salida de manejo clinico"].value_counts()

# replace 'seguimiento'
df["Estado del paciente a la salida de manejo clinico"].replace({"Seguimiento": "Recuperado"}, inplace=True)
df["Estado del paciente a la salida de manejo clinico"].value_counts()


#------------------------------------------------------------------------------



# drop rows with nan

#df.dropna()
df.dropna(inplace = True)
df.info()

dff = df.copy()

# integer encoding
cleanup_nums = {"Sexo":     {"Masculino": 0, "Femenino": 1},
                "Estancia en UCI": {"No": 0, "Sí": 1},
                "Estado del paciente a la salida de manejo clinico": {"Recuperado": 1, "Fallecido": 0}}

dff = dff.replace(cleanup_nums)

dff.info()

# convert every column to categorical except for one

cols=[i for i in df.columns if i not in ["Edad"]]
for col in cols:
    dff[col] = dff[col].astype('category')

# -----------------------------------------------------------------------------

dff.info()
# FEATURE SELECTION

# categorical feature selection via chi-squared test

# getting all the categorical columns except the target
categorical_columns = dff.select_dtypes(exclude = 'number').drop('Estado del paciente a la salida de manejo clinico', axis = 1).columns

dff.info()

# import the function
from scipy.stats import chi2_contingency

# try the test with every categorical quesiton
chi2_check = []
for i in categorical_columns:
    if chi2_contingency(pd.crosstab(dff['Estado del paciente a la salida de manejo clinico'], dff[i]))[1] < 0.05:
        chi2_check.append('Reject Null Hypothesis')
    else:
        chi2_check.append('Fail to Reject Null Hypothesis')
        res = pd.DataFrame(data = [categorical_columns, chi2_check] 
             ).T
        
res.columns = ['Column', 'Hypothesis']
print(res)


#-----------------------------------------------------------------------------

# POST HOC TESTING

check = {}
for i in res[res['Hypothesis'] == 'Reject Null Hypothesis']['Column']:
    dummies = pd.get_dummies(dff[i])
    bon_p_value = 0.05/dff[i].nunique()
    for series in dummies:
        if chi2_contingency(pd.crosstab(dff['Estado del paciente a la salida de manejo clinico'], dummies[series]))[1] < bon_p_value:
            check['{}-{}'.format(i, series)] = 'Reject Null Hypothesis'
        else:
            check['{}-{}'.format(i, series)] = 'Fail to Reject Null Hypothesis'

res_chi_ph = pd.DataFrame(data = [check.keys(), check.values()]).T
res_chi_ph.columns = ['Pair', 'Hypothesis']
res_chi_ph



#------------------------------------------------------------------------------

# BORUTA

x = dff.iloc[:, np.r_[0:43]]
y = dff.iloc[:, 43]


### make X_shadow by randomly permuting each column of X
np.random.seed(42)
X_shadow = x.apply(np.random.permutation)
X_shadow.columns = ['shadow_' + feat for feat in x.columns]### make X_boruta by appending X_shadow to X
X_boruta = pd.concat([x, X_shadow], axis = 1)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

### fit a random forest
forest = RandomForestClassifier(max_depth = 5, random_state = 42)
forest.fit(X_boruta, y)### store feature importances
feat_imp_X = forest.feature_importances_[:len(x.columns)]
feat_imp_shadow = forest.feature_importances_[len(x.columns):]### compute hits
hits = feat_imp_X > feat_imp_shadow.max()

#-----------------------------------------------------------------------------

# one-hot encoding requeried

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np###initialize Boruta
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 5
)
boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 100 # number of trials to perform
)### fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(x), np.array(y))### print results
green_area = x.columns[boruta.support_].to_list()
blue_area = x.columns[boruta.support_weak_].to_list()

print('features in the green area:', green_area)
print('features in the blue area:', blue_area)




#------------------------------------------------------------------------------

# Correspondence analysis















#-------------------------------------------------------------------------------

df2 = df.copy()
df2.loc[df['Edad'] < 70, 'Edad'] = 5
df2.loc[df['Edad'] >= 70, 'Edad'] = 0

