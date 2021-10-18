# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 05:02:06 2021

@author: User
"""


#---------------------#
# DEMOGRAFICOS
#---------------------#

import pandas as pd
import numpy as np

path = "C:\\Users\\User\\Desktop\\Modelo UniAndes"

df = pd.read_excel(path + '\\output\\demo_clean.xlsx')


# remove excel created columns and paciente
df = df.drop(df.columns[[0,1]], axis = 1)

# transform ventilacion to numeric
cleanup_col = {"Ventilacion":     {"Invasiva": 1, "No invasiva": 0}}
df = df.replace(cleanup_col)

# convert entire data set to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# impute data because of nas
from sklearn.impute import SimpleImputer

# define imputer
imputer = SimpleImputer(strategy='median')

# fit on the dataset
imputer.fit(df)


# transform the dataset
df2 = pd.DataFrame(imputer.transform(df))

df2.columns = df.columns

# METHOD 1: PRINCIPAL COMPONENT ANALYSIS

# features
X= df2.drop(columns=["Ventilacion"])
# objective variable
y= df2['Ventilacion']

from sklearn.decomposition import PCA
model=PCA(n_components=10).fit(X)
n_pcs = model.components_.shape[0]
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
df.columns[most_important]


# FEATURES
# Index(['Dificultad respiratoria si disnea y si taquipnea', 'Hipertension',
#        'Fiebre', 'Edad', 'Malestar General', 'Malestar General', 'Diabetes',
#        'Tos', 'Cefalea', 'Tos'],
#       dtype='object')

#------------------------------------------------------------------------------

# METHOD 2: BORUTA

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

train = pd.get_dummies(df2, drop_first=False, dummy_na=True)
train.shape

features = [f for f in train.columns if f not in ['Ventilacion']]
len(features)

X = train[features].values
Y = train['Ventilacion'].values.ravel()

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter = 50, perc = 90)
boruta_feature_selector.fit(X, Y)

X_filtered = boruta_feature_selector.transform(X)
X_filtered.shape

final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(features[x])
print(final_features)

# FEATURES
# ['Disnea', 'Diabetes', 'Otra enfermedad cardiaca']

#------------------------------------------------------------------------------


#---------------------#
# SIGNOS VITALES
#---------------------#


# Read data
sv2 = pd.read_excel(path + '\\output\\signos_vitales_clean.xlsx')

sv2 = sv2.drop(sv2.columns[[0]], axis = 1)


# return ventilacion to numeric
cleanup_col = {"Ventilacion":     {"Invasiva": 1, "No invasiva": 0}}
sv2 = sv2.replace(cleanup_col)

# convert entire data set to numeric
sv2 = sv2.apply(pd.to_numeric, errors='coerce')

# impute data because of nas
from sklearn.impute import SimpleImputer

# define imputer
imputer = SimpleImputer(strategy='median')

# fit on the dataset
imputer.fit(sv2)


# transform the dataset
sv_f = pd.DataFrame(imputer.transform(sv2))

sv_f.columns = sv2.columns

# features
X= sv_f.drop(columns=["Ventilacion"])
# objective variable
y= sv_f['Ventilacion']


# PRINCIPAL COMPONENT ANALYSIS
from sklearn.decomposition import PCA
model=PCA(n_components=7).fit(X)
n_pcs = model.components_.shape[0]
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
sv_f.columns[most_important]

# FEATURES
# ['TA SISTOLICA', 'FC', 'TA DIASTOLICA', 'FR', 'OXI', 'TEMP', 'TAM']

#------------------------------------------------------------------------------

# BORUTA

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

train = pd.get_dummies(sv_f, drop_first=False, dummy_na=True)
train.shape

features = [f for f in train.columns if f not in ['Ventilacion']]
len(features)


X = train[features].values
Y = train['Ventilacion'].values.ravel()


rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter = 50, perc = 90)
boruta_feature_selector.fit(X, Y)

X_filtered = boruta_feature_selector.transform(X)
X_filtered.shape


final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(features[x])
print(final_features)


#['FC', 'TA SISTOLICA', 'TA DIASTOLICA', 'TAM', 'TEMP', 'OXI', 'FR']

#------------------------------------------------------------------------------


#---------------------#
# PARACLINICOS
#---------------------#


# Read data
pc2sub = pd.read_excel(path + '\\output\\paraclinicos_clean.xlsx')

# remove date column
pc2sub = pc2sub.drop(pc2sub.columns[[0]], axis = 1)


# return ventilacion to numeric
cleanup_col = {"Ventilacion":     {"Invasiva": 1, "No invasiva": 0}}
pc2sub = pc2sub.replace(cleanup_col)

# convert entire data set to numeric
pc2sub = pc2sub.apply(pd.to_numeric, errors='coerce')

# impute data because of nas
from sklearn.impute import SimpleImputer

# define imputer
imputer = SimpleImputer(strategy='median')

# fit on the dataset
imputer.fit(pc2sub)


# transform the dataset
pc_f = pd.DataFrame(imputer.transform(pc2sub))

pc_f.columns = pc2sub.columns


# features
X= pc_f.drop(columns=["Ventilacion"])
# objective variable
y= pc_f['Ventilacion']



# PRINCIPAL COMPONENT ANALYSIS
from sklearn.decomposition import PCA
model=PCA(n_components=11).fit(X)
n_pcs = model.components_.shape[0]
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
pc_f.columns[most_important]


# FEATURES
#['Urea', 'FR', 'Linfocitos', 'FC', 'D', 'Plaquetas', 'Edad', 'S', 'PO2', 'FiO2']

#------------------------------------------------------------------------------

# BORUTA

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

train = pd.get_dummies(pc_f, drop_first=False, dummy_na=True)
train.shape

features = [f for f in train.columns if f not in ['Ventilacion']]
len(features)


X = train[features].values
Y = train['Ventilacion'].values.ravel()


rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter = 50, perc = 90)
boruta_feature_selector.fit(X, Y)

X_filtered = boruta_feature_selector.transform(X)
X_filtered.shape


final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for x in np.nditer(indexes):
    final_features.append(features[x])
print(final_features)

# FEATURES
#['Linfocitos', 'Urea', 'Creatinina', 'Edad', 'D']



