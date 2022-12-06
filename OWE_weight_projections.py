# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 18:00:32 2022

@author: Benitez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
import statistics
import os
import unicodedata
import csv

#Folder with cvs files: D:\Benitez\Offshore_LCA\Paper_publication\Paper_about_wind_offshore\CSVs_files

#Location of weight components of a 2MW wind turbine:
#path = r'D:\Benitez\Offshore_LCA\Paper_publication\Paper_about_wind_offshore'
path_b = r'D:\Benitez\Offshore_LCA\Paper_publication\Paper_about_wind_offshore\CSVs_files\Scalation_factor'
path_dia = r'D:\Benitez\Offshore_LCA\Paper_publication\Paper_about_wind_offshore\CSVs_files\Rotor_diameter'
path_wei = r'D:\Benitez\Offshore_LCA\Paper_publication\Paper_about_wind_offshore\CSVs_files\Weight_evol'

def read_b (path_b):
    
    ''' csv contains scalation factor "b" per wind turbine compoenent
        Source: Table 4; Caduff, M., et al. (2012). "Wind power electricity: 
        the bigger the turbine, the greener the electricity?" 
        Environmental Science & Technology 46(9): 4725-4733.

        path_b = r'D:\Benitez\Offshore_LCA\Paper_publication\
                Paper_about_wind_offshore\Scalation_factor'
    '''
    files = os.listdir(path_b) 
    '''reading csv files'''
    for f in files:
        if os.path.splitext(f)[1] == ".csv":
            df = pd.read_csv( path_b + '\\' + f )    
    return df

def read_dia (path_dia):
    
    ''' csv contains rotor diameters in m for wind turbines of nominal
    capacity 6, 7, 8, 9.5, 11 and 15 MW

        path_dia = r'D:\Benitez\Offshore_LCA\Paper_publication\
                    Paper_about_wind_offshore\Rotor_diameter'
    '''
    files = os.listdir(path_dia) 
    '''reading csv files'''
    for f in files:
        if os.path.splitext(f)[1] == ".csv":
            df = pd.read_csv( path_dia + '\\' + f )    
    return df

def read_weights_owe (path):
    
    ''' csv contains weight in ton of componenets of a 2 MW wind turbine over 
        the years 2022 to 2050
        values in ton
        path = r'D:\Benitez\Offshore_LCA\Paper_publication\
                    Paper_about_wind_offshore\CSVs_files'
    '''
    
    files = os.listdir(path)
    '''reading csv files'''
    for f in files:
        if os.path.splitext(f)[1] == ".csv":
            df = pd.read_csv( path + '\\' + f )    
    return df

owe_weights = read_weights_owe (path)
type(owe_weights['2022'])

dia_std = df_dia.describe().iloc[[2]]  #standar deviation rotor diameters

'''scaling components'''
dia_average = df_dia.iloc[[1]]  # average rotor diameters in meters
tmp = wg.set_index(['Component'])
components = ['Foundation', 'Tower', 'Rotor', 'Nacelle']
MW = ['2.00', '3.75', '5.00', '6.50', '8.00', '9.50', '11.00', '15.00']

tmp1 = []
tmp2 = []
for c in components:
    for m in MW:
        print(c,m)
        comp_mean = tmp.loc[c, '2022':'2050'].describe().loc[['mean']].rename(index={'mean': c + str('_') + m + str('_MW')})
        
        tmp1.append(comp_mean)
        tmp2 = pd.concat(tmp1)
        
        comp_std = tmp.loc[c, '2022':'2050'].describe().loc[['std']].rename(index={'std': c + str('_std')})
        b_comp = b.drop(['relationship'],1).set_index(['Component']).T
        b_mean = b_comp[c].describe().loc[['mean']]
        b_std = b_comp[c].describe().loc[['std']]
        comp_mean.append(test, ignore_index=False)
        
        
        
        
        cp = comp_mean.mul(pow(dia_average[m][1]/76, b_mean[0]),0)
        ind = cp.index[0]
        cp = cp.rename(index={ind: ind + str('_')+ m})
        comp_wg.append(cp)
    
    test = test.drop(['b'], axis=1)
    test.columns += ' ' + str(d) + '_MW'
    mass.append(test)
    MW_mass = pd.concat(mass, axis=1)

MW_mass.columns




#mass foundation 2 MW in 2022 max: 1755 ton and min 1439 ton
years = ['2022', '2025', '2030', '2035', '2040', '2045', '2050']
low = [1439, 1382, 1239,	 1184, 1132, 1104, 1086] # weight projection 2Mw foundation best scnenario
high = [1755, 1755, 1755	, 1618,	1558, 1526,	1502]
MW = [6, 7,	8,	10,	11,	15]
rotor_diameter_max = [166, 169,	180, 191, 220, 234]
rotor_dia_min = [136, 139, 148, 157, 180, 192]


'''example to filter dataframe: owe_weights[owe_weights.Scenarios == 'Pessimistic'][owe_weights.Component == 'Foundation']
'''

''' To obtain the dataframe df: contains weights component between a max and mix value, 2022 to 2050'''

components = ['Foundation', 'Tower', 'Rotor', 'Nacelle']
#to name the columns
name_col = []
for c in components:
    for y in years:
        col = (str(c) + '_' + str(y))
        name_col.append(col)
  
#to obtain df_t
comp = []
for c in components:
    print(c)
    high = owe_weights[owe_weights.Scenarios == 'Pessimistic'][owe_weights.Component == c]
    low = owe_weights[owe_weights.Scenarios == 'Optimistic'][owe_weights.Component == c]
    high = high.loc[:, '2022':'2050'].values.tolist()
    low = low.loc[:, '2022':'2050'].values.tolist()
    high = high[0]
    low = low[0]
    
    for l, h in zip(low, high):
        print(l, h)
        np.random.seed(3)
        x = np.random.randint(low = l, high = h, size = 1000)
        comp.append(x)
        weights = np.transpose(np.array(comp))
    
df_t = pd.DataFrame(data=weights, columns=name_col)

#Stadistic analysis
stadistic_2MW = df_t.describe()
  
def results_to_csv (df, file_name, path):
    
    '''
    weight development in tons per component of a 2 MW wind turbine
    ''' 
    df.to_csv(path + '\\' + file_name + '.csv')
    return file_name + ' ' + '.csv' + ' ' + 'saved in ' + path

#Stadistic analysis saved in cvs:
results_to_csv(stadistic_2MW, 'weights_development_2MW',  path)

#Scalation components

# random scaling factor (b) or caduff coef, sample=1000
#foundation b = 1.20 to 2.09
np.random.seed(1)
b = np.random.uniform(low = 1.20, high = 2.09, size = 1000)

b = read_b (path_b)
b_coef = []
for c in components:
    b_high = b[b.Component==c].b_high.values.tolist()[0]
    b_low = b[b.Component==c].b_low.values.tolist()[0]
    #for l, h in zip(b_low, b_high, size=1000):
    np.random.seed(1)
    coef = np.random.uniform(low =  b_low, high = b_high, size = 1000)
    b_coef.append(coef)
    b_scaling_fac = np.transpose(np.array(b_coef))

b_scaling_factors = pd.DataFrame(data=b_scaling_fac, columns=components)
#Stadistic analysis
stadistic_b = b_scaling_factors.describe()


data = []
for i in range(len(low)):
    for j in range(len(low[0])):
        for l, h in zip(low[i], high[i]):
            np.random.seed(3)
            x = np.random.randint(low = l, high = h, size = 1000)
            data.append(x)
            weights = np.transpose(np.array(data))
            
df = pd.DataFrame(data=weights,columns=years)

df['b'] = b

tmp = []
for r_max, r_min in zip(rotor_diameter_max, rotor_dia_min):
    np.random.seed(4)
    test = np.random.randint(low = r_min, high = r_max, size = 1000)
    tmp.append(test)
    rotors = np.transpose(np.array(tmp))
    
df_dia = pd.DataFrame(data=rotors,columns=MW)
df_dia.insert(0, 2, 76)


mass = []
MW_mass = []
for d in MW:
    test = df_wg.mul(pow(df_dia[d]/76,df_b['b']),0)
    test = test.drop(['b'], axis=1)
    test.columns += ' ' + str(d) + '_MW'
    mass.append(test)
    MW_mass = pd.concat(mass, axis=1)

MW_mass.columns


names = ['6_MW', '7_MW', '8_MW', '10_MW', '11_MW', '15_MW' ]
for n in names:
    n
    name = [col for col in MW_mass.columns if n in col]
    nw = MW_mass.filter(name).describe()
    #MW_mass.filter(name)
    #mean = MW_mass.filter(name).mean()
    nw.loc[['mean','std']]
    type (info)
    
    statistics.mean(MW_mass.filter(name))
    
 
    x_col = np.sort(MW_mass.filter(name).to_numpy())

col_year = []
year_mean = []
year_sd = []
for col in MW_mass.columns:
    x_col = np.sort(MW_mass[col].to_numpy())
    mean_col = statistics.mean(x_col)
    sd_col = statistics.stdev(x_col)
   # col_year.append(x_col)
    plt.plot(x_col, norm.pdf(x_col, mean_col, sd_col))
plt.show()





x_array = np.sort(MW_mass['2022 6_MW'].to_numpy())
x2_array = np.sort(MW_mass['2025 6_MW'].to_numpy())
mean = statistics.mean(x_array)
sd = statistics.stdev(x_array)

mean2 = statistics.mean(x2_array)
sd2 = statistics.stdev(x2_array)

plt.plot(x_array, norm.pdf(x_array, mean, sd))
plt.plot(x2_array, norm.pdf(x2_array, mean2, sd2))
plt.xlabel('Foundation weights, ton')
plt.ylabel('Probability')
plt.title('Normal distrubution')
plt.text(6500, .00025, r'$\mu_{2022} =5000,\ \sigma=1059$')
plt.text(6500, .000225, r'$\mu_{2025} =4910,\ \sigma=1072$')
#plt.xlim(40, 160)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()




plt.hist(x_array, bins=50, density=True, alpha=0.6, color='b')
plt.hist(x2_array, bins=50, density=True, alpha=0.6, color='r')
plt.xlabel('Foundation weights, ton')
plt.ylabel('Probability')
plt.title('Histogram of foundation weight 6 MW')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.xlim(40, 160)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()




np.random.seed(2)
rotor = np.random.randint(low = 136, high = 166, size = 1000)

random_float_list = []
# Set a length of the list to 10
for i in range(0, 1000):
    
    # any random float between 50.50 to 500.50
    # don't use round() if you need number as it is
    x = round(random.uniform(1.20, 2.09), 2)
    x
    random_float_list.append(x)

print(random_float_list)
# Out
coef_caduf = random_float_list

mass_2MW = 1439 #kg
Rotor_6MW = 166 #m
Rotor_2MW = 76 #m

mass_found_float_list = []
for coef in coef_caduf:
    x = mass_2MW* pow(Rotor_6MW/Rotor_2MW, coef)
    mass_found_float_list.append(x)

print(mass_found_float_list)