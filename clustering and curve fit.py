# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:14:41 2023

@author: umamah
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# For Suppressing warnings
import warnings
warnings.filterwarnings('ignore')



from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler

import scipy.optimize as opt

plt.style.use("ggplot")


def file(l):
    '''function to call the both files of worldbank'''
    
    D = l[['Country Name','Indicator Code','Indicator Name','1990','1991','1992','1993','1994','1995',
           '1996','1997','1998','1999','2000','2001','2002','2003','2004',
           '2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']]
   
    return D


#variables to call the worldbank data files 

gdp = pd.read_csv('C:/Users/samkh/Desktop/applied 03/gdp_percapita.csv',error_bad_lines=False)
co2 = pd.read_csv('C:/Users/samkh/Desktop/applied 03/co2.csv',error_bad_lines=False)


#function call to call the files for further implementation

gdp_df=file(gdp)
co2_df=file(co2)
co2_df.head(30)

#Clean up process Dataset for relavant information
#using MEAN statistical function to fill NAN values witht the mean values
Countriesgdp = gdp_df[(gdp_df['Indicator Code'] == 'NY.GDP.PCAP.KD.ZG')]
Countriesgdp.reset_index()
Countriesgdp.fillna(Countriesgdp.mean(numeric_only=True).round(1), inplace=True)
Countriesgdp

Countries_co2 = co2_df[(co2_df['Indicator Code'] == 'EN.ATM.CO2E.PC')]
Countries_co2.reset_index()
Countries_co2.fillna(Countries_co2.mean(numeric_only=True).round(1), inplace=True)

#extracting dataframe for gdp and co2 production of all the countries of worldbank in the year 1990 

CO2andGDP1990 = Countriesgdp[['Country Name','1990']]
CO2andGDP1990.rename(columns = {'1990':'GDPin1990'}, inplace = True)
CO2andGDP1990['CO2in1990'] = Countries_co2[['1990']]
print(CO2andGDP1990)

##extracting dataframe for gdp and co2 production of all the countries of worldbank in the year 2015 
CO2andGDP2015 = Countriesgdp[['Country Name','2015']]
CO2andGDP2015.rename(columns = {'2015':'GDPin2015'}, inplace = True)
CO2andGDP2015['CO2in2015'] = Countries_co2[['2015']]
CO2andGDP2015.head()






#feature scaling to  normalize the values to a specific range for good clusters
#scaling the values to normalize it into a specific range


scale = MinMaxScaler()
scale.fit(CO2andGDP1990[['GDPin1990']])
CO2andGDP1990['GDPin1990'] = scale.transform(CO2andGDP1990[['GDPin1990']])

scale.fit(CO2andGDP1990[['CO2in1990']])
CO2andGDP1990['CO2in1990'] = scale.transform(CO2andGDP1990[['CO2in1990']])


print(CO2andGDP1990)


#scaling the values of 2015 dataframe to normalize it into a specific range


scale = MinMaxScaler()
scale.fit(CO2andGDP2015[['GDPin2015']])
CO2andGDP2015['GDPin2015'] = scale.transform(CO2andGDP2015[['GDPin2015']])

scale.fit(CO2andGDP2015[['CO2in2015']])
CO2andGDP2015['CO2in2015'] = scale.transform(CO2andGDP2015[['CO2in2015']])


print(CO2andGDP2015)


#using elbow method to determine the optimal value of K for K-means clustering
x = CO2andGDP1990['CO2in1990']
y = CO2andGDP1990['GDPin1990']



data = list(zip(x, y))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method for 1990')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


#using elbow method to determine the optimal value of K for K-means clustering
x = CO2andGDP2015['CO2in2015']
y = CO2andGDP2015['GDPin2015']



data = list(zip(x, y))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method for 2015')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#we got K=3 clusters as an optimal value
#implementing kmeans clustering with no.of clusters=3
km = KMeans(n_clusters=3)
predicted = km.fit_predict(CO2andGDP1990[['CO2in1990','GDPin1990']])
print(predicted)

#finding the centres of the clusters
center = km.cluster_centers_
print(center)

#adding clustering id to the dataframe to see which country is in which cluster
country_clustered1990 = CO2andGDP1990.iloc[:,:]
country_clustered1990 = pd.concat([CO2andGDP1990, pd.DataFrame(predicted, columns=['cluster'])], axis = 1)
country_clustered1990.head(10)

#silhouette score v/s no.clusters for 1990

del CO2andGDP1990['Country Name']
labels = km.labels_

ssd = []
num_of_clusters = list(range(2,10))
silhouette_value = []
for n in range(2,10):
    km = KMeans(n_clusters = n, random_state=101).fit(CO2andGDP1990)
    silhouette_value.append(silhouette_score(CO2andGDP1990, km.labels_))

fig,ax = plt.subplots()    
plt.plot(num_of_clusters, silhouette_value, marker='X', label=silhouette_value)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Number of Clusters vs. Silhouette Score")

plt.show()

#silhouette score v/s no.clusters for 2015
del CO2andGDP2015['Country Name']
labels = km.labels_

ssd = []
num_of_clusters = list(range(2,10))
silhouette_value = []
for n in range(2,10):
    km = KMeans(n_clusters = n, random_state=101).fit(CO2andGDP2015)
    silhouette_value.append(silhouette_score(CO2andGDP2015, km.labels_))

fig,ax = plt.subplots()    
plt.plot(num_of_clusters, silhouette_value, marker='X', label=silhouette_value)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Number of Clusters vs. Silhouette Score for dataframe of 2015")

plt.show()

#printing silhouette_score of the Dataframe of 1990
print('SILHOUETTE_SCORE OFco2andgdp1990',silhouette_score(CO2andGDP1990,km.labels_))

#printing silhouette_score of the Dataframe of 2015
print('SILHOUETTE_SCORE OFco2andgdp2015',silhouette_score(CO2andGDP2015,km.labels_))


#we got K=3 clusters as an optimal value
#implementing kmeans clustering with no.of clusters=3 on 2015 dataframe to compare both clusters 
km = KMeans(n_clusters=3)
predicted_values = km.fit_predict(CO2andGDP2015[['CO2in2015','GDPin2015']])
print(predicted_values)

#finding the centres of the clusters of 2015 dataframe
cen=km.cluster_centers_
print(cen)

#adding clustering id to the dataframe to see which country is in which cluster
country_clustered2015 = CO2andGDP2015.iloc[:,:]
country_clustered2015 = pd.concat([CO2andGDP2015, pd.DataFrame(predicted_values, columns=['cluster'])], axis = 1)
country_clustered2015.head(10)


#plotting clusters graphically to visualise

df0 = country_clustered1990[country_clustered1990.cluster==0]
df1 = country_clustered1990[country_clustered1990.cluster==1]
df2 = country_clustered1990[country_clustered1990.cluster==2]

df3 = country_clustered2015[country_clustered2015.cluster==0]
df4 = country_clustered2015[country_clustered2015.cluster==1]
df5 = country_clustered2015[country_clustered2015.cluster==2]


plt.scatter(df0.CO2in1990,df0['GDPin1990'],color='red',s=20,label='cluster0')
plt.scatter(df1.CO2in1990,df1['GDPin1990'],color='green',s=20,label='cluster1')
plt.scatter(df2.CO2in1990,df2['GDPin1990'],color='blue',s=20,label='cluster2')
plt.scatter(center[:,0],center[:,1],color='yellow',s=80,label='centriods')
plt.title('Countries Clusters in 1990[CO2v/s GDP]')
plt.xlabel('CO2 of the Countries')
plt.ylabel('GDP of the Countries')
plt.legend()
plt.show()

plt.scatter(df3.CO2in2015,df3['GDPin2015'],color='orange',s=10,label='cluster0')
plt.scatter(df4.CO2in2015,df4['GDPin2015'],color='yellow',s=10,label='cluster1')
plt.scatter(df5.CO2in2015,df5['GDPin2015'],color='magenta',s=10,label='cluster2')
plt.scatter(cen[:,0],cen[:,1],color='black',s=80,label='centriods')
plt.title('Countries Clusters in 2015[CO2 v/s GDP]')
plt.xlabel('CO2 of the Countries')
plt.ylabel('GDP of the Countries')
plt.legend()
plt.show()

