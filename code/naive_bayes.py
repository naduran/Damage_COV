# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 06:28:32 2021

@author: User
"""

# import libraries
import pandas as pd
import numpy as np
#from tabulate import tabulate
# machine learning framework
import h2o
#h2o.init(nthreads = -1, max_mem_size = 12)
h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()
# shutdown per every trained model in case of crash
#h2o.cluster().shutdown() 

path = "C:\\Users\\User\\Desktop\\Modelo UniAndes"

# read data
df = pd.read_excel(path + '\\output\\demo_clean.xlsx')

# remove excel created columns
df = df.drop(df.columns[[0]], axis = 1)

# return ventilacion to numeric (naive bayes needs all data to be numeric in h2o)
cleanup_col = {"Ventilacion":     {"Invasiva": 1, "No invasiva": 0, 6:1}}
df = df.replace(cleanup_col)

cleanup_col = {"Ninguno":     {4: 1}}
df = df.replace(cleanup_col)

# convert entire data set to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# # also, there can't be nas in all the data set
# # impute data because of nas
# from sklearn.impute import SimpleImputer

# # define imputer
# imputer = SimpleImputer(strategy='median')

# # fit on the dataset
# imputer.fit(df)

# # transform the dataset
# df2 = pd.DataFrame(imputer.transform(df))

# df2.columns = df.columns

# #df2.isnull().values.any()

from h2o.estimators import H2ONaiveBayesEstimator


# no feature selection
#---------------------#
# MODEL 1: SINTOMAS
#---------------------#



# create features list
y = 'Ventilacion'
#y = df.columns[-1]

x = list(df.columns[1:21])
#del x[-1]

df_m = df.iloc[:,np.r_[1:21,53:54],]

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

# convert every column to factor
df_m = df_m.asfactor()


#nb.train(x=x, y=y, training_frame=df_m)


# predict
#nb_predictions = nb.predict(test_data=df_m)


# print performance metrics
# nb_performance = nb.model_performance()
# print(nb_performance)


#------------------------------------------------------------------------------

# no feature selection
#---------------------#
# MODEL 2: COMORBILIDADES
#---------------------#


# create features list
y = 'Ventilacion'
#y = df.columns[-1]

x = list(df.columns[1:21])
#del x[-1]

df_m = df.iloc[:,np.r_[2:3,22:55],]

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

# convert every column to factor
df_m = df_m.asfactor()


#nb.train(x=x, y=y, training_frame=df_m)


# predict
#nb_predictions = nb.predict(test_data=df_m)


# print performance metrics
#nb_performance = nb.model_performance()
#print(nb_performance)



#------------------------------------------------------------------------------

h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()
# shutdown per every trained model in case of crash
#h2o.cluster().shutdown()

# feature selection: PCA
#---------------------#
# MODEL 3: SINTOMAS + COMORBILIDADES
#---------------------#


# create features list
y = 'Ventilacion'
#y = df.columns[-1]


selected_features = ['Dificultad respiratoria si disnea y si taquipnea', 'Hipertension', 'Fiebre', 'Edad', 'Malestar General', 'Diabetes', 'Cefalea', 'Tos']
all_variables = ['Dificultad respiratoria si disnea y si taquipnea', 'Hipertension', 'Fiebre',
                 'Edad', 'Malestar General', 'Diabetes', 'Cefalea', 'Tos', 'Ventilacion']
x = list(df[selected_features])
#del x[-1]

#df_m = df.iloc[:,np.r_[2:3,1:55],]
df_m = df[all_variables]

df_m.columns = df_m.columns.str.strip()


df_m.dtypes

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

# convert every column to factor (very important for h2o training)
df_m = df_m.asfactor()

df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]


# naive bayes estimator: very importance to set laplace=1 otherwise could encounter null exception
nb = H2ONaiveBayesEstimator(model_id='sincom', laplace=1)

# train model
nb.train(x=x, y=y, training_frame=train, validation_frame=valid)

# predict
nb_predictions = nb.predict(test_data=df_m)


# print performance metrics
nb_performance = nb.model_performance()
print(nb_performance)



#------------------------------------------------------------------------------


h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()
# shutdown per every trained model in case of crash
#h2o.cluster().shutdown()

# feature selection: Both PCA and Boruta
#---------------------#
# MODEL 3: SIGNOS VITALES
#---------------------#


# Read data
sv2 = pd.read_excel(path + '\\output\\signos_vitales_clean.xlsx')

sv2 = sv2.drop(sv2.columns[[0]], axis = 1)

# clean response variable
sv2.loc[sv2['Ventilacion'] == 6, 'Ventilacion'] = "Invasiva"
sv2['Ventilacion'].unique()

# create features list
y = 'Ventilacion'
#y = df.columns[-1]


selected_features = ['TA SISTOLICA', 'FC', 'TA DIASTOLICA', 'FR', 'OXI', 'TEMP', 'TAM']
all_variables = ['TA SISTOLICA', 'FC', 'TA DIASTOLICA', 'FR', 'OXI', 'TEMP', 'TAM', 'Ventilacion']
x = list(sv2[selected_features])
#del x[-1]

#df_m = df.iloc[:,np.r_[2:3,1:55],]
df_m = sv2[all_variables]

df_m.columns = df_m.columns.str.strip()


df_m.dtypes

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

# convert response variable column to factor (very important for h2o training)
df_m['Ventilacion'] = df_m['Ventilacion'].asfactor()

df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# naive bayes estimator: very importance to set laplace=1 otherwise could encounter null exception
nb = H2ONaiveBayesEstimator(model_id='sincom', laplace=1)

# train model
nb.train(x=x, y=y, training_frame=train, validation_frame=valid)

# predict
nb_predictions = nb.predict(test_data=df_m)

# print performance metrics
nb_performance = nb.model_performance()
print(nb_performance)



#------------------------------------------------------------------------------


h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()
# shutdown per every trained model in case of crash
#h2o.cluster().shutdown()

# feature selection: Both PCA and Boruta
#---------------------#
# MODEL 3: PARACLINICOS
#---------------------#

# Read data
pc2sub = pd.read_excel(path + '\\output\\paraclinicos_clean.xlsx')

# remove date column
pc2sub = pc2sub.drop(pc2sub.columns[[0]], axis = 1)

pc2sub['Ventilacion'].unique()

# create features list
y = 'Ventilacion'
#y = df.columns[-1]


selected_features = ['Urea', 'FR', 'Linfocitos', 'FC', 'D', 'Plaquetas', 'Edad', 'S', 'PO2', 'FiO2']
all_variables = ['Urea', 'FR', 'Linfocitos', 'FC', 'D', 'Plaquetas', 'Edad', 'S', 'PO2', 'FiO2', 'Ventilacion']
x = list(pc2sub[selected_features])
#del x[-1]

#df_m = df.iloc[:,np.r_[2:3,1:55],]
df_m = pc2sub[all_variables]

df_m.columns = df_m.columns.str.strip()


df_m.dtypes

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

# convert every column to factor (very important for h2o training)
df_m = df_m.asfactor()

df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# naive bayes estimator: very importance to set laplace=1 otherwise could encounter null exception
nb = H2ONaiveBayesEstimator(model_id='sincom', laplace=1)

# train model
nb.train(x=x, y=y, training_frame=train, validation_frame=valid)

# predict
nb_predictions = nb.predict(test_data=df_m)

# print performance metrics
nb_performance = nb.model_performance()
print(nb_performance)
