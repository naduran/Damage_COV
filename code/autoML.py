# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 01:42:48 2021

@author: User
"""

# AUTO MACHINE LEARNING

#------------------------------------------------------------------------------

#---------------------#
# MODEL 1: SINTOMAS
#---------------------#


import pandas as pd
import numpy as np
import os

os.chdir("..")
path = os.path.abspath(os.getcwd())


df = pd.read_excel(path + '\\output\\demo_clean.xlsx')
print("df 1")
print(df)

print ("Modelo 1: Sintomas")
import h2o
from tabulate import tabulate
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='16G')
h2o.connect()

# convert to h2o frame
df_m = df.iloc[:,np.r_[1:20,54:55],]
print("demo_clean_select 1")
# print(df_m)
df_m = h2o.H2OFrame(df_m)

splits = df_m.split_frame(ratios=[0.8], seed=1)
train = splits[0]
test = splits[1]

# features
y = "Ventilacion"
x = df_m.columns
print(x)
x.remove(y)

# train automl
aml = H2OAutoML(max_runtime_secs=120, seed=1)
aml.train(x=x,y=y, training_frame=train)

lb = aml.leaderboard
print(lb.head())
lb.as_data_frame().head()

print(tabulate(lb.as_data_frame().head(10), headers=['index','modelid','auc','logloss','mean_per_class_error', 'rmse'], tablefmt='psql'))

pred = aml.predict(test)
pred.as_data_frame().head()
h2o.cluster().shutdown()

#------------------------------------------------------------------------------

# remove accent in columns
df = df.rename(columns={'Lupus eritematoso sistémico': 'Lupus eritematoso sistemico'})
df = df.rename(columns={'Síndrome de Sjögren': 'Sindrome de Sjogren'})

#---------------------#
# MODEL 2: COMORBILIDADES
#---------------------#

print("Modelo 2: Comoribilidades")
h2o.init(max_mem_size='16G')
h2o.connect()
# convert to h2o frame
df_m = df.iloc[:,np.r_[2:3,22:55],]
df_m = h2o.H2OFrame(df_m)

splits = df_m.split_frame(ratios=[0.8],seed=1)
train = splits[0]
test = splits[1]

# features
y = "Ventilacion"
x = df_m.columns
x.remove(y)

# train automl
aml = H2OAutoML(max_runtime_secs=120, seed=1)
aml.train(x=x,y=y, training_frame=train)

lb = aml.leaderboard

print(tabulate(lb.as_data_frame().head(50), headers=['index','modelid','auc','logloss','mean_per_class_error', 'rmse'], tablefmt='psql'))

pred = aml.predict(test)
pred.as_data_frame().head()

h2o.cluster().shutdown

#------------------------------------------------------------------------------



# remove accent
df = df.rename(columns={'Lupus eritematoso sistémico': 'Lupus eritematoso sistemico'})
df = df.rename(columns={'Síndrome de Sjögren': 'Sindrome de Sjogren'})

#---------------------#
# MODEL 3: SINTOMAS + COMORBILIDADES
#---------------------#

print("Modelo 3: Sintomas + Comoribilidades")
# convert to h2o frame
h2o.init(max_mem_size='16G')
h2o.connect()
df_m = df.iloc[:,np.r_[2:55],]
df_m = h2o.H2OFrame(df_m)

splits = df_m.split_frame(ratios=[0.8],seed=1)
train = splits[0]
test = splits[1]

# features
y = "Ventilacion"
x = df_m.columns
x.remove(y)

# train automl
aml = H2OAutoML(max_runtime_secs=120, seed=1)
aml.train(x=x,y=y, training_frame=train)

lb = aml.leaderboard

print(tabulate(lb.as_data_frame().head(50), headers=['index','modelid','auc','logloss','mean_per_class_error', 'rmse'], tablefmt='psql'))

pred = aml.predict(test)
pred.as_data_frame().head()
h2o.cluster().shutdown





#-----------------------------------------------------------------------------


#---------------------#
# MODEL 5: PARACLINICOS
#---------------------#

print("Modelo 5: Paraclinicos")
# Read data
h2o.init(max_mem_size='16G')
h2o.connect()
pc2sub = pd.read_excel(path + '\\output\\paraclinicos_clean.xlsx')

# remove date column
pc2sub = pc2sub.drop(pc2sub.columns[[0]], axis = 1)


# convert to h2o frame
df_m = pc2sub.iloc[:,np.r_[0:14],]
df_m = h2o.H2OFrame(df_m)


splits = df_m.split_frame(ratios=[0.8],seed=1)
train = splits[0]
test = splits[1]

# features
y = "Ventilacion"
x = df_m.columns
x.remove(y)

# train automl
aml = H2OAutoML(max_runtime_secs=120, seed=1)
aml.train(x=x,y=y, training_frame=train)

lb = aml.leaderboard

print(tabulate(lb.as_data_frame().head(50), headers=['index','modelid','auc','logloss','mean_per_class_error', 'rmse'], tablefmt='psql'))

pred = aml.predict(test)
pred.as_data_frame().head()





