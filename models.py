# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:23:05 2021

@author: User
"""

# IMPORT H2O
import pandas as pd
import numpy as np
#import matplotlib
from tabulate import tabulate
import h2o
h2o.init(nthreads = -1, max_mem_size = 8)
h2o.connect()
#h2o.cluster().shutdown() 

path = "C:\\Users\\User\\Desktop\\Modelo UniAndes"

# Read data
df = pd.read_excel(path + '\\output\\demo_clean.xlsx')

df.dtypes

# remove excel created columns
df = df.drop(df.columns[[0]], axis = 1)

# select every column except for "Paciente"
cols=[i for i in df.columns if i not in ["Paciente"]]
#cols=[i for i in df.columns]

for col in cols:
    df[col] = df[col].astype('category')

# check types again
df.dtypes

#------------------------------------------------------------------------------

# RANDOM FOREST

# import random forest
from h2o.estimators.random_forest import H2ORandomForestEstimator
#rf = H2ORandomForestEstimator(seed=1)
#rf = H2ORandomForestEstimator(seed=1, balance_classes=True)

assignment_type = "Stratified"

rf = H2ORandomForestEstimator(fold_assignment = assignment_type,seed=1,balance_classes=True,categorical_encoding="auto",
                              ntrees = 200,
                              max_depth=20,
                              nfolds=10)

#------------------------------------------------------------------------------

# MODEL 1: SINTOMAS

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

# FIT AND PREDICT
rf.train(x=x, y=y, training_frame=train, validation_frame=valid)

y_hat = rf.predict(test_data=test)

y_hat.as_data_frame()

# PERFORMANCE EVALUATION

rf_performance = rf.model_performance(test)
print(rf_performance)

#rf_performance.auc()
#rf_performance.confusion_matrix()

# find importance
rf.varimp(True)


# hyperparameter optimization with grid search

#h2o.shutdown()
from h2o.grid import H2OGridSearch

drf_hyper_params = {
                "ntrees" : [10,25,50],
                "max_depth": [ 5, 7, 10],
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

rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)


len(rf_grid)


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

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

best_drf_model.model_id

# predict
drf_predictions = best_drf_model.predict(test_data=test)


# performance
best_drf_model.score_history()

print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))



best_drf_model.varimp_plot()

best_drf_model.plot(metric="auc")

grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)

# compare
y_hat.as_data_frame()
drf_predictions.as_data_frame()


###############################################################################
###############################################################################

# remove accent
df = df.rename(columns={'Lupus eritematoso sistémico': 'Lupus eritematoso sistemico'})
df = df.rename(columns={'Síndrome de Sjögren': 'Sindrome de Sjogren'})



# MODEL 2: COMORBILIDADES

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

# FIT AND PREDICT
rf.train(x=x, y=y, training_frame=train, validation_frame=valid)

y_hat = rf.predict(test_data=test)

y_hat.as_data_frame()

# PERFORMANCE EVALUATION

rf_performance = rf.model_performance(test)
print(rf_performance)

#rf_performance.auc()
#rf_performance.confusion_matrix()

# find importance
rf.varimp(True)


# Hyperparameter optimization
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

rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)


len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

best_drf_model.model_id

# predict
drf_predictions = best_drf_model.predict(test_data=test)


# performance
best_drf_model.score_history()

best_drf_model.varimp()
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))

best_drf_model.varimp_plot()

best_drf_model.plot(metric="auc")

grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)


###############################################################################
###############################################################################

# remove accent
df = df.rename(columns={'Lupus eritematoso sistémico': 'Lupus eritematoso sistemico'})
df = df.rename(columns={'Síndrome de Sjögren': 'Sindrome de Sjogren'})


# MODEL 3: SINTOMAS + COMORBILIDADES

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

# FIT AND PREDICT
rf.train(x=x, y=y, training_frame=train, validation_frame=valid)

y_hat = rf.predict(test_data=test)

y_hat.as_data_frame()

# PERFORMANCE EVALUATION

rf_performance = rf.model_performance(test)
print(rf_performance)

#rf_performance.auc()
#rf_performance.confusion_matrix()

# find importance
rf.varimp(True)


# Hyperparameter optimzation
rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)


len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

best_drf_model.model_id

# predict
drf_predictions = best_drf_model.predict(test_data=test)


# performance
best_drf_model.score_history()

best_drf_model.varimp()
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))


best_drf_model.varimp_plot()

best_drf_model.plot(metric="auc")

grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)


###############################################################################
###############################################################################

# MODEL 4: SIGNOS VITALES

# Read data
sv = pd.read_excel(path + '\\output\\signos_vitales.xlsx')

sv = sv.drop(sv.columns[[0]], axis = 1)

# remove white spaces at both ends
#sv.columns = sv.columns.str.strip()

sv['Paciente'] = sv['Paciente'].str.replace('Paciente ', '')

dfsub = df.iloc[:,np.r_[0:1,53:54],]

sv['Paciente']=sv['Paciente'].astype(int)

# join dataframes
sv2 = sv.merge(dfsub, on='Paciente', how='left')

sv2.dtypes
dfsub.dtypes

# drop Paciente and TM columns
sv2 = sv2.drop(sv2.columns[[0,3]], axis = 1)

sv2.dtypes

# separate TA column
sv2[['TA SISTOLICA', 'TA DIASTOLICA']] = sv2['TA'].str.split('/', 1, expand=True)
sv2 = sv2.drop(sv2.columns[[1]], axis = 1)



# convert TEMP to int
sv2["TEMP"] = sv2["TEMP"].str.replace(',', '.')

sv2["TEMP"] = pd.to_numeric(sv2["TEMP"])

sv2 = sv2.dropna()

sv2["TA SISTOLICA"] = pd.to_numeric(sv2["TA SISTOLICA"])
sv2["TA DIASTOLICA"] = pd.to_numeric(sv2["TA DIASTOLICA"])

# create column based on formula
#TAM = (S+2*D)/3

sv2["TAM"] = (sv2["TA SISTOLICA"] + (2 * sv2["TA DIASTOLICA"]) ) / 3

sv2 = sv2[['FC', 'TA SISTOLICA', 'TA DIASTOLICA', 'TAM', 'TEMP', 'OXI', 'FR', 'Ventilacion']]

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

#df_m.anyfactor()
df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# FIT AND PREDICT
rf.train(x=x, y=y, training_frame=train, validation_frame=valid)

y_hat = rf.predict(test_data=test)

y_hat.as_data_frame()

# PERFORMANCE EVALUATION

rf_performance = rf.model_performance(test)
print(rf_performance)

#rf_performance.auc()
#rf_performance.confusion_matrix()

# find importance
rf.varimp(True)


# hyperparameter optimization with grid search
rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)


len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

best_drf_model.model_id

# predict
drf_predictions = best_drf_model.predict(test_data=test)


# performance
best_drf_model.score_history()

best_drf_model.varimp()
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))

best_drf_model.varimp_plot()

best_drf_model.plot(metric="auc")

grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)

###############################################################################
###############################################################################

# MODEL 5: PARACLINICOS

# Read data
pc = pd.read_excel(path + '\\input\\paraclinicos.xlsx')

# remove date column
pc = pc.drop(pc.columns[[14]], axis = 1)

#
dfsub = df.iloc[:,np.r_[0:1,53:54],]

# remove white spaces
pc.columns = pc.columns.str.strip()

pc['Paciente'] = pc['Paciente'].astype(int)


# join dataframes
pc2 = pc.merge(dfsub, on='Paciente', how='left')

# not na y variable 
pc2 = pc2[pc2['Ventilacion'].notna()]

# drop columns with too many nas
pc2 = pc2.drop(pc2.columns[[0,3,4,5,6,7]], axis = 1)

pc2.dtypes

# data without y variable so we can impute missing data
pc2sub = pc2.drop(pc2.columns[[13]], axis = 1)

#data imputation


from sklearn.impute import SimpleImputer

# define imputer
imputer = SimpleImputer(strategy='median')

# fit on the dataset
imputer.fit(pc2sub)


# transform the dataset
pc2sub = pd.DataFrame(imputer.transform(pc2sub)) 

pc2sub["Ventilacion"] = pc2["Ventilacion"]

pc2sub.columns = list(pc2.columns)[0:14]

# Variable	Rango menor 	Rango mayor 
# Po2	              80	100
# Plaquetas	          150	450
# Linfocitos	      0,9	4,52
# Bilirrubina	        0	1
# Urea	                8	23
# Creatinina	     0,51	0,95
# Troponina 	        0	14
# LDH	               60	160
# Dimero d	            0	300
# Ferritinina	        0	300
# Creatinina	      0,7	1,3

# encode by normal values

pc2sub.dtypes

pc2sub.loc[pc2sub.PO2.between(80, 100), "PO2"] = 1
pc2sub.loc[pc2sub['PO2'] !=1, 'PO2'] = 0

pc2sub.loc[pc2sub.Plaquetas.between(150, 450), "Plaquetas"] = 1
pc2sub.loc[pc2sub['Plaquetas'] !=1, 'Plaquetas'] = 0

pc2sub.loc[pc2sub.Linfocitos.between(0.9, 4.52), "Linfocitos"] = 1
pc2sub.loc[pc2sub['Linfocitos'] !=1, 'Linfocitos'] = 0

pc2sub.loc[pc2sub.Bilirubina.between(0, 1), "Bilirubina"] = 1
pc2sub.loc[pc2sub['Bilirubina'] !=1, 'Bilirubina'] = 0

pc2sub.loc[pc2sub.Urea.between(8, 23), "Urea"] = 1
pc2sub.loc[pc2sub['Urea'] !=1, 'Urea'] = 0

pc2sub.loc[pc2sub.Creatinina.between(0.51, 0.95), "Creatinina"] = 1
pc2sub.loc[pc2sub['Creatinina'] !=1, 'Creatinina'] = 0

pc2sub.loc[pc2sub.S.between(90, 140), "S"] = 1
pc2sub.loc[pc2sub['S'] !=1, 'S'] = 0

pc2sub.loc[pc2sub.D.between(60, 90), "D"] = 1
pc2sub.loc[pc2sub['D'] !=1, 'D'] = 0

pc2sub.loc[pc2sub.FR.between(12, 20), "FR"] = 1
pc2sub.loc[pc2sub['FR'] !=1, 'FR'] = 0

pc2sub.loc[pc2sub.FC.between(60, 100), "FC"] = 1
pc2sub.loc[pc2sub['FC'] !=1, 'FC'] = 0

pc2sub.loc[pc2sub['Edad'] < 70, 'Edad'] = 1
pc2sub.loc[pc2sub['Edad'] >= 70, 'Edad'] = 0


# temporarily remove columns without normal ranges
pc2sub = pc2sub.drop(pc2sub.columns[[5,12]], axis = 1)


#------------------------------------------------------------------------------

rf = H2ORandomForestEstimator(fold_assignment = assignment_type,seed=1,balance_classes=True,categorical_encoding="auto",
                              ntrees = 200,
                              max_depth=20,
                              nfolds=5)



# create features list
y = 'Ventilacion'

#x = list(df.columns[1:3])
x = list(pc2sub.iloc[:,np.r_[0:11]].columns)
#del x[-1]

df_m = pc2sub.iloc[:,np.r_[0:12],]

# convert to h2o frame
df_m = h2o.H2OFrame(df_m)

#df_m["Ventilacion"] = df_m["Ventilacion"].asfactor()

#df_m.anyfactor()
df_m.types

# split into train, validation and test
splits = df_m.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# FIT AND PREDICT
rf.train(x=x, y=y, training_frame=train, validation_frame=valid)

y_hat = rf.predict(test_data=test)

y_hat.as_data_frame()

# PERFORMANCE EVALUATION

rf_performance = rf.model_performance(test)
print(rf_performance)

#rf_performance.auc()
#rf_performance.confusion_matrix()

# find importance
rf.varimp(True)


# hyperparamter optimization
rf_grid.train(x=x, y=y, training_frame=train, validation_frame=valid)


len(rf_grid)

# best model
best_drf_model = find_best_model_from_grid(rf_grid, "auc")

best_drf_model.model_id

# predict
drf_predictions = best_drf_model.predict(test_data=test)


# performance
best_drf_model.score_history()

best_drf_model.varimp()
print(tabulate(pd.DataFrame(best_drf_model.varimp()),headers=['variable','relative importance','scaled importance','percentage'], tablefmt='psql'))

best_drf_model.varimp_plot()

best_drf_model.plot(metric="auc")

grid_rf_performance = best_drf_model.model_performance(test)
print(grid_rf_performance)

