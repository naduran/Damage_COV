# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 00:43:11 2021

@author: User
"""

import pandas as pd
import numpy as np

path = "C:\\Users\\User\\Desktop\\Modelo UniAndes"

df = pd.read_excel(path + '\\output\\demo_clean.xlsx')

# Read data
pc = pd.read_excel(path + '\\input\\paraclinicos.xlsx')

# remove date column
pc = pc.drop(pc.columns[[15]], axis = 1)

# select 'paciente' and 'ventilacion' columns to match each patient in 'paraclinicos' data
dfsub = df.iloc[:,np.r_[1:2,54:55],]

# remove white spaces
pc.columns = pc.columns.str.strip()

# keep until Paciente 165, row index 664 for now
pc = pc.iloc[np.r_[0:665],:]

pc['Paciente'] = pc['Paciente'].astype(int)


# join dataframes
pc2 = pc.merge(dfsub, on='Paciente', how='left')

# not na y variable 
pc2 = pc2[pc2['Ventilacion'].notna()]

# percentage of missings per column
pc2.isnull().sum() * 100 / len(pc2)

# drop columns with too many nas (leaving Bilirubina for now since descarting it makes a drastic change)
pc2sub = pc2.drop(pc2.columns[[0,3,4,5,6,7,9]], axis = 1)




# # data imputation (this step may not be necessary for random forest but it is for base model)

# # data without response variable so we can impute missing data
# pc2sub = pc2.drop(pc2.columns[[14]], axis = 1)

# from sklearn.impute import SimpleImputer

# # define imputer
# imputer = SimpleImputer(strategy='median')

# # fit on the dataset
# imputer.fit(pc2sub)

# # transform the dataset
# pc2sub = pd.DataFrame(imputer.transform(pc2sub)) 

# pc2sub["Ventilacion"] = pc2["Ventilacion"]

# pc2sub.columns = list(pc2.columns)[0:15]




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
# pH                  7.35  7.45
# FR                  12    20
# FC                  60    100
# S                   90    140
# D                   60    90
# FiO2                0.19  0.21
# Glasgov             11    13

# encode by normal values and ignoring na values

pc2sub['Creatinina'].dtype

pc2sub.loc[pc2sub.PO2.between(80, 100, inclusive = True), "PO2"] = 1
pc2sub['PO2'] = np.where(pc2sub['PO2'] > 1 | (pc2sub['PO2'] < 1), 0, pc2sub['PO2'])

pc2sub.loc[pc2sub.Plaquetas.between(150, 450, inclusive = True), "Plaquetas"] = 1
pc2sub['Plaquetas'] = np.where(pc2sub['Plaquetas'] > 1 | (pc2sub['Plaquetas'] < 1), 0, pc2sub['Plaquetas'])

pc2sub.loc[pc2sub.Linfocitos.between(0.9, 4.52, inclusive = True), "Linfocitos"] = 1
pc2sub['Linfocitos'] = np.where((pc2sub['Linfocitos'] > 1) | (pc2sub['Linfocitos'] < 1), 0, pc2sub['Linfocitos'])

#pc2sub.loc[pc2sub.Bilirubina.between(0, 1), "Bilirubina"] = 1
#pc2sub['Bilirubina'] = np.where(pc2sub['Bilirubina'] > 1 | (pc2sub['Linfocitos'] < 1), 0, pc2sub['Bilirubina'])

pc2sub.loc[pc2sub.Urea.between(8, 23, inclusive = True), "Urea"] = 1
pc2sub['Urea'] = np.where(pc2sub['Urea'] > 1 | (pc2sub['Urea'] < 1), 0, pc2sub['Urea'])

pc2sub.loc[pc2sub.Creatinina.between(0.51, 0.95, inclusive = True), "Creatinina"] = 1
pc2sub['Creatinina'] = np.where(pc2sub['Creatinina'] > 1, 0, pc2sub['Creatinina'])
pc2sub['Creatinina'] = np.where(pc2sub['Creatinina'] < 1, 0, pc2sub['Creatinina'])

pc2sub.loc[pc2sub.S.between(90, 140, inclusive = True), "S"] = 1
pc2sub['S'] = np.where(pc2sub['S'] > 1 | (pc2sub['S'] < 1), 0, pc2sub['S'])

pc2sub.loc[pc2sub.D.between(60, 90, inclusive = True), "D"] = 1
pc2sub['D'] = np.where(pc2sub['D'] > 1 | (pc2sub['D'] < 1), 0, pc2sub['D'])

pc2sub.loc[pc2sub.FR.between(12, 20, inclusive = True), "FR"] = 1
pc2sub['FR'] = np.where(pc2sub['FR'] > 1 | (pc2sub['FR'] < 1), 0, pc2sub['FR'])

pc2sub.loc[pc2sub.FC.between(60, 100, inclusive = True), "FC"] = 1
pc2sub['FC'] = np.where(pc2sub['FC'] > 1 | (pc2sub['FC'] < 1), 0, pc2sub['FC'])

pc2sub.loc[pc2sub['Edad'] < 70, 'Edad'] = 1
pc2sub.loc[pc2sub['Edad'] >= 70, 'Edad'] = 0

pc2sub.loc[pc2sub.FC.between(7.35, 7.45, inclusive = True), "pH"] = 1
pc2sub['pH'] = np.where(pc2sub['pH'] > 1, 0, pc2sub['pH'])
pc2sub['pH'] = np.where(pc2sub['pH'] < 1, 0, pc2sub['pH'])

pc2sub.loc[pc2sub.FC.between(0.19, 0.21, inclusive = True), "FiO2"] = 1
pc2sub['FiO2'] = np.where((pc2sub['FiO2'] < 1), 0, pc2sub['FiO2'])

pc2sub.loc[pc2sub.FC.between(11, 13, inclusive = True), "Glasgov"] = 1
pc2sub['Glasgov'] = np.where(pc2sub['Glasgov'] > 1 | (pc2sub['Glasgov'] < 1), 0, pc2sub['Glasgov'])


# remove nas in response variable
pc2sub = pc2sub[pc2sub['Ventilacion'].notna()]

# verify only two classes
pc2sub['Ventilacion'].unique()

pc2sub.to_excel(path + "\\output\\paraclinicos_clean.xlsx", sheet_name='sheet1')
