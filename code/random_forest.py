# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:23:05 2021

@author: User
"""

def find_best_model_from_grid(h2o_grid, test_parameter):    
    model_list = []
    for grid_item in h2o_grid:
        if test_parameter == "r2":
            if not (grid_item.r2() == "NaN"):
                model_list.append(grid_item.r2())
            else:
                model_list.append(0.0)            
        elif test_parameter == "auc":
            if not (grid_item.auc() == "NaN"):
                model_list.append(grid_item.auc())
            else:
                model_list.append(0.0)            
    #print(model_list)        
    max_index = model_list.index(max(model_list))
    #print(max_index)
    best_model = h2o_grid[max_index]
    print("Model ID with best R2: " +  best_model.model_id)
    if test_parameter == "r2":
        print("Best R2: " +  str(best_model.r2()))
    elif test_parameter == "auc":
        print("Best AUC: " +  str(best_model.auc()))
    return best_model

import h2o

# import random forest
from h2o.estimators.random_forest import H2ORandomForestEstimator

# stratified data set
assignment_type = "Stratified"

from h2o.grid import H2OGridSearch

drf_hyper_params = {
                "ntrees" : [10,25,50],
                "max_depth": [ 5, 7, 10],
                "sample_rate": [0.5, 0.75, 1.0]}

grid_search_criteria = {"strategy": "RandomDiscrete", 
                        "max_models": 50, 
                        "seed": 12345}


rf_grid = H2OGridSearch(model=H2ORandomForestEstimator(
                                                        seed=1,
                                                        nfolds=5,
                                                        fold_assignment=assignment_type,
                                                        balance_classes = True,
                                                        categorical_encoding = "auto",
                                                        keep_cross_validation_predictions=True),
                     hyper_params=drf_hyper_params,
                     search_criteria=grid_search_criteria,
                     grid_id="rf_grid")

# import libraries
import pandas as pd
import numpy as np
from tabulate import tabulate
# machine learning framework
h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()
# shutdown per every trained model
#h2o.cluster().shutdown() 

path = "C:\\Users\\User\\Desktop\\Modelo UniAndes"

# read data
df = pd.read_excel(path + '\\output\\demo_clean.xlsx')

# remove excel created columns
df = df.drop(df.columns[[0]], axis = 1)

cleanup_col = {"Ninguno":     {4: 1}}
df = df.replace(cleanup_col)

# check types again
df.dtypes

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------

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

#df_m["Ventilacion"] = df_m["Ventilacion"].asfactor()

# convert every column to factor
df_m = df_m.asfactor()

#df_m.anyfactor()
df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# hyperparameter optimization with grid search

#h2o.shutdown()


rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)

len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

# predict
drf_predictions = best_drf_model.predict(test_data=test)

# performance
best_drf_model.score_history()

# print variable importance table
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))

# variable importance plot
best_drf_model.varimp_plot()

# print performance metrics
grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)


#------------------------------------------------------------------------------

# remove accent in columns
df = df.rename(columns={'Lupus eritematoso sistémico': 'Lupus eritematoso sistemico'})
df = df.rename(columns={'Síndrome de Sjögren': 'Sindrome de Sjogren'})

#---------------------#
# MODEL 2: COMORBILIDADES
#---------------------#


# create features list
y = 'Ventilacion'

# change name of 

#x = list(df.columns[1:3])
x = list(df.iloc[:,np.r_[1:3,21:53]].columns)
#del x[-1]

df_m = df.iloc[:,np.r_[1:3,21:54],]

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

#df_m["Ventilacion"] = df_m["Ventilacion"].asfactor()

# convert every column to factor
df_m = df_m.asfactor()

#df_m.anyfactor()
df_m.types


# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]


rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)

len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

# predict
drf_predictions = best_drf_model.predict(test_data=test)

# performance
best_drf_model.score_history()

# variable importance table
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))

# variable importance plot
best_drf_model.varimp_plot()

# performance metrics table
grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)


#------------------------------------------------------------------------------

# shutdown per every trained model and connect again
#h2o.cluster().shutdown()
h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()



# remove accent
df = df.rename(columns={'Lupus eritematoso sistémico': 'Lupus eritematoso sistemico'})
df = df.rename(columns={'Síndrome de Sjögren': 'Sindrome de Sjogren'})

#---------------------#
# MODEL 3: SINTOMAS + COMORBILIDADES
#---------------------#

# create features list
y = 'Ventilacion'

#x = list(df.columns[1:3])
x = list(df.iloc[:,np.r_[1:53]].columns)
#del x[-1]

df_m = df.iloc[:,np.r_[1:54],]

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

#df_m["Ventilacion"] = df_m["Ventilacion"].asfactor()

# convert every column to factor
df_m = df_m.asfactor()

#df_m.anyfactor()
df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]


# Hyperparameter optimzation
rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)

len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

# predict
drf_predictions = best_drf_model.predict(test_data=test)

# performance
best_drf_model.score_history()

# variable importance table
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))

importance_demo = pd.DataFrame(best_drf_model.varimp())
importance_demo.columns =['Variable', 'Importancia relativa', 'Importance escalada', 'Porcentaje']

# save
importance_demo.to_excel(path + "\\output\\demo_importance.xlsx", sheet_name='sheet1')

# variable importance plot
best_drf_model.varimp_plot()

# performance metrics table
grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)



#------------------------------------------------------------------------------


# shutdown per every trained model and connect again
#h2o.cluster().shutdown()
h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()


#---------------------#
# MODEL 4: SIGNOS VITALES
#---------------------#


# Read data
sv2 = pd.read_excel(path + '\\output\\signos_vitales_clean.xlsx')

sv2 = sv2.drop(sv2.columns[[0]], axis = 1)

# clean response variable
sv2.loc[sv2['Ventilacion'] == 6, 'Ventilacion'] = "Invasiva"
sv2['Ventilacion'].unique()
#------------------------------------------------------------------------------

# create features list
y = 'Ventilacion'

#x = list(df.columns[1:3])
x = list(sv2.iloc[:,np.r_[0:7]].columns)
#del x[-1]

df_m = sv2.iloc[:,np.r_[0:8],]

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

#df_m["Ventilacion"] = df_m["Ventilacion"].asfactor()

# here the features are numeric
df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# need more complex trees for this amount of data
drf_hyper_params = {
                "ntrees" : [10,25,50,100],
                "max_depth": [ 5, 7, 10, 20],
                "sample_rate": [0.5, 0.75, 1.0]}

grid_search_criteria = {"strategy": "RandomDiscrete", 
                        "max_models": 100, 
                        "seed": 12345}


rf_grid = H2OGridSearch(model=H2ORandomForestEstimator(
                                                        seed=1,
                                                        nfolds=5,
                                                        fold_assignment=assignment_type,
                                                        balance_classes = True,
                                                        categorical_encoding = "auto",
                                                        keep_cross_validation_predictions=True),
                     hyper_params=drf_hyper_params,
                     search_criteria=grid_search_criteria,
                     grid_id="rf_grid")


# hyperparameter optimization with grid search
rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)

len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

# predict
drf_predictions = best_drf_model.predict(test_data=test)

# performance
best_drf_model.score_history()

# variable importance table
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))

importance_signos = pd.DataFrame(best_drf_model.varimp())
importance_signos.columns =['Variable', 'Importancia relativa', 'Importance escalada', 'Porcentaje']

# save
importance_signos.to_excel(path + "\\output\\signos_importance.xlsx", sheet_name='sheet1')

# variable importance plot
best_drf_model.varimp_plot()

# performance metrics table
grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)




#------------------------------------------------------------------------------



# shutdown per every trained model and connect again
#h2o.cluster().shutdown()
h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()


#---------------------#
# MODEL 5: PARACLINICOS
#---------------------#


# Read data
pc2sub = pd.read_excel(path + '\\output\\paraclinicos_clean.xlsx')

# remove date column
pc2sub = pc2sub.drop(pc2sub.columns[[0]], axis = 1)


#------------------------------------------------------------------------------

# create features list
y = 'Ventilacion'

#x = list(df.columns[1:3])
x = list(pc2sub.iloc[:,np.r_[0:13]].columns)
#del x[-1]

df_m = pc2sub.iloc[:,np.r_[0:14],]

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

#df_m["Ventilacion"] = df_m["Ventilacion"].asfactor()

# convert every column to factor
df_m = df_m.asfactor()

#df_m.anyfactor()
df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# back to standard grid search
drf_hyper_params = {
                "ntrees" : [10,25,50],
                "max_depth": [ 5, 7, 10],
                "sample_rate": [0.5, 0.75, 1.0]}

grid_search_criteria = {"strategy": "RandomDiscrete", 
                        "max_models": 50, 
                        "seed": 12345}


rf_grid = H2OGridSearch(model=H2ORandomForestEstimator(
                                                        seed=1,
                                                        nfolds=5,
                                                        fold_assignment=assignment_type,
                                                        balance_classes = True,
                                                        categorical_encoding = "auto",
                                                        keep_cross_validation_predictions=True),
                     hyper_params=drf_hyper_params,
                     search_criteria=grid_search_criteria,
                     grid_id="rf_grid")

# hyperparamter optimization
rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)

len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

# predict
drf_predictions = best_drf_model.predict(test_data=test)

# variable importance table
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))

importance_paraclinicos = pd.DataFrame(best_drf_model.varimp())
importance_paraclinicos.columns =['Variable', 'Importancia relativa', 'Importance escalada', 'Porcentaje']

# save
importance_paraclinicos.to_excel(path + "\\output\\paraclinicos_importance.xlsx", sheet_name='sheet1')

# variable importance plot
best_drf_model.plot(metric="auc")

# performance metrics table
grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)


