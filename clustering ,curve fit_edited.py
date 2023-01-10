# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:14:41 2023

@author: umamah
"""

#importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

from scipy.optimize import curve_fit
from numpy import arange

plt.style.use("ggplot")

#user defined functions used in the coding
def file(l):
    '''function to call the both files of worldbank'''
    
    D = l[['Country Name','Indicator Code','Indicator Name','1990','1991','1992','1993','1994','1995',
           '1996','1997','1998','1999','2000','2001','2002','2003','2004',
           '2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']]
   
    return D


def mapping_function(x,a,b,c):
    '''function for mapping the values for curve fitting'''
    comp = a * x + b * x**2 + c
    return comp



#main code
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


#plotting the countries as a representation of CO2 v/s GDP in 1990 and 2015 before clustering

fig,ax = plt.subplots(2,figsize = (6,10))

ax[0].scatter(CO2andGDP1990['CO2in1990'],CO2andGDP1990['GDPin1990'],color = 'black',s=15, label='Countries')
ax[0].set_title('Countries graph as GDP and CO2(1990)')
ax[0].set_xlabel('CO2')
ax[0].set_ylabel('GDP')
ax[0].legend(loc='upper left')


ax[1].scatter(CO2andGDP2015['CO2in2015'],CO2andGDP2015['GDPin2015'],color = 'blue',s=15,label='Countries')
ax[1].set_title('Countries graph as GDP and CO2(2015)')
ax[1].set_xlabel('CO2')
ax[1].set_ylabel('GDP')
ax[1].legend(loc='upper left')

fig.tight_layout()
plt.savefig('without cluster countries.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()



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
plt.savefig('elbow method 1990.png', format='png',dpi=600,
            bbox_inches='tight')
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
plt.savefig('elbow method 2019.png', format='png',dpi=600,
            bbox_inches='tight')
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
print(country_clustered2015)


#plotting clusters graphically to visualise

df0 = country_clustered1990[country_clustered1990.cluster==0]
df1 = country_clustered1990[country_clustered1990.cluster==1]
df2 = country_clustered1990[country_clustered1990.cluster==2]

df3 = country_clustered2015[country_clustered2015.cluster==0]
df4 = country_clustered2015[country_clustered2015.cluster==1]
df5 = country_clustered2015[country_clustered2015.cluster==2]



fig,ax = plt.subplots(2,figsize = (12,15))
bbox=dict(boxstyle="round", alpha=0.1)
arrowprops=dict(arrowstyle='<-' ,  connectionstyle="arc3,rad=-0.3")

ax[0].scatter(df0.CO2in1990,df0['GDPin1990'],color='yellow',s=20,label='cluster1')
ax[0].scatter(df1.CO2in1990,df1['GDPin1990'],color='pink',s=20,label='cluster2')
ax[0].scatter(df2.CO2in1990,df2['GDPin1990'],color='blue',s=20,label='cluster3')
ax[0].scatter(center[:,0],center[:,1],color='black',s=80,label='centriods')
ax[0].set_title('Countries Clusters in 1990[CO2v/s GDP]')
ax[0].set_xlabel('CO2 of the Countries')
ax[0].set_ylabel('GDP of the Countries')
ax[0].legend()


ax[0].annotate('CHINA', xy=(0.064718,0.243038),
            arrowprops=arrowprops , bbox = bbox, fontsize=15, weight='bold')


ax[0].annotate('CANADA', xy=(0.512347,0.228792),
            arrowprops=arrowprops , bbox = bbox, fontsize=15, weight='bold')

ax[0].annotate('USA', xy=(0.656382,0.219472),
            arrowprops=arrowprops, bbox = bbox, fontsize=15, weight='bold')
            

ax[0].annotate('South.A' , xy=(0.210022,0.161710),
            arrowprops=arrowprops, bbox = bbox, fontsize=15, weight='bold')

ax[0].annotate('UK',xy=(0.331869,0.215107),
            arrowprops=arrowprops, bbox = bbox,fontsize=15, weight='bold')

ax[0].annotate('IND' , xy=(0.021861,0.255653),
            arrowprops=arrowprops, bbox = bbox, fontsize=15, weight='bold')


ax[1].scatter(df3.CO2in2015,df3['GDPin2015'],color='orange',s=10,label='cluster1')
ax[1].scatter(df4.CO2in2015,df4['GDPin2015'],color='red',s=10,label='cluster2')
ax[1].scatter(df5.CO2in2015,df5['GDPin2015'],color='magenta',s=10,label='cluster3')

ax[1].scatter(cen[:,0],cen[:,1],color='black',s=80,label='centriods')
ax[1].set_title('Countries Clusters in 2015[CO2 v/s GDP]')
ax[1].set_xlabel('CO2 of the Countries')
ax[1].set_ylabel('GDP of the Countries')
ax[1].legend()

ax[1].annotate('CHINA', xy=(0.202687,0.684125),
            arrowprops=arrowprops , bbox = bbox, fontsize=15, weight='bold')


ax[1].annotate('CAN', xy=(0.445090,0.561578),
            arrowprops=arrowprops , bbox = bbox, fontsize=14.5, weight='bold')

ax[1].annotate('USA', xy=(0.442565,0.600022),
            arrowprops=arrowprops, bbox = bbox, fontsize=14.5, weight='bold')
            

ax[1].annotate('South.A' , xy=(0.215694,0.548991),
            arrowprops=arrowprops, bbox = bbox, fontsize=15, weight='bold')

ax[1].annotate('UK',xy=(0.174547,0.593094),
            arrowprops=arrowprops, bbox = bbox,fontsize=15, weight='bold')

ax[1].annotate('IND' , xy=(0.045447,0.689778),
            arrowprops=arrowprops, bbox = bbox, fontsize=15, weight='bold')

fig.tight_layout()
plt.savefig('clustered countries.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()


#visualising the centres of both the years in the graph to see 
#the difference pf GDP and CO2 values in the past 25 years

x = center[:,0],center[:,1]
y = cen[:,0],cen[:,1]


plt.plot(center[:,0],center[:,1],'--',color='grey',label='joining line')
plt.plot(cen[:,0],cen[:,1],'--',color='grey')

plt.scatter(center[:,0],center[:,1],color='red',s=200,label='cluster centres of 1990')
plt.scatter(cen[:,0],cen[:,1],color='black',s=200,label='cluster centres of 2015')

plt.title('evolution of the cluster centers (1990 and 2015)')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 emission')
plt.legend(loc='center')
plt.savefig('centers clusters.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()


#CURVE FITT ON THE GDP PER CAPITA 
#implementing curve fit function on the GDP of the country Australia
#for extracting GDP of Australia of all the last 58 years


GDPofAus = gdp[gdp['Country Name']=='Australia']
del GDPofAus['Country Name']
del GDPofAus['Indicator Name']
del GDPofAus['Country Code']
del GDPofAus['Indicator Code']
del GDPofAus['1960']
print(type(GDPofAus))
gdparr = GDPofAus.values.tolist()
print(gdparr)

#making dataframe of aus gdps for all 58 years
year = []
for i in range(59):
    year.append(1961+i)


x = []
for i in range(59):
    x.append(i)

dfaus = pd.DataFrame(columns = ['years','gdp'],
                    index = x )
print(dfaus.loc[0][0])
for i in range(59):
    dfaus.loc[i] = [year[i],gdparr[0][i]]
dfaus.head(60)


#plt.scatter(dfusa['years'],dfusa['gdp'],color = 'red',s=20,label = 'gdpvalues of USA')
dfaus.plot('years','gdp',color='red')
plt.title('GDP of Australia from past 58 years')
plt.xlabel('Year from 1960 to 2019')
plt.ylabel('GDP per capita')
plt.legend()
plt.savefig('gdp usa.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()

x = dfaus['years']
y = dfaus['gdp']
#curve fit
opt_param,covar = curve_fit(mapping_function,x,y)

#summarize the parametric values
a, b, c = opt_param
#plot input v/s output
plt.scatter(x, y, color='green',
           label='data')

#defining the sequence of inputs between smallest and largest known inputs
x_line = arange(min(x),max(x),1)
#calculate the output for the range
y_line = mapping_function(x_line,a,b,c)

#create a line graph for mapping function
plt.plot(x_line,y_line,'--',color='red',label='curvefitted line')
plt.xlabel('ground truths')
plt.ylabel('prediction')
plt.title('curve fitting on GDP')
plt.legend(loc='upper right',prop={'size':8})
plt.savefig('curve fit.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()

sigma = np.sqrt(np.diag(covar))
print("parameters:", opt_param)
print("std. dev.", sigma)

#forecasting for next ten years
year = np.arange(1960, 2031)
print(year)
forecast = mapping_function(year, *opt_param)
print(forecast)

plt.figure()
plt.scatter(dfaus["years"], dfaus["gdp"], label="GDP")
plt.plot(year, forecast, label="forecast",color='green')
plt.title('curve fit functioning for next 10 years')
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.savefig('curve fit forecaste.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()





