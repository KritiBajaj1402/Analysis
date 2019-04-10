#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
os.chdir("C:/Users/130901/Desktop/HousingData")
train = pd.read_csv("train.csv")
print(train.columns)


# In[2]:


train['SalePrice'].describe()


# In[4]:


sns.distplot(train['SalePrice'])


# In[5]:


###Positive skewdnees Mean>MEDIAN>MODE and more houses have saleprice more than 200000

##using karl pearson
##mean - mode/standard deviation

##kutosis measures the peakness

print("Skewnees")
print(train['SalePrice'].skew())
print("kutosis")
print(train['SalePrice'].kurt())


# In[6]:


## Lets check the vary with the different variable
plt.scatter(train['SalePrice'],train['GrLivArea'])
plt.show()


# In[9]:



plt.scatter(train['SalePrice'],train['TotalBsmtSF'])
plt.show()


# In[18]:


print(train['OverallQual'].unique())


# In[21]:


sns.boxplot(x = "OverallQual", y = "SalePrice",data = train)


# In[23]:


###with the increase in overall  qual saleprice also increases and quality 10 has a higher IQR
plt.subplots(figsize = (16,8))
sns.boxplot(x = "YearBuilt", y = "SalePrice",data = train)


# In[32]:


## from this figure we can say the newly built houses are more costly

##'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. 
#Both relationships are positive, which means that as one variable increases, the other 
#also increases. In the case of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.

## from this figure we can say the newly built houses are more costly
##'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. 
#Both relationships are positive, which means that as one variable increases, the other 
#also increases. In the case of 'TotalBsmtSF', we can see that the slope of the 
#linear relationship is particularly high.


##lets build the correlation matrix

correlation_matrix = train.corr()
plt.subplots(figsize = (16,8))
sns.heatmap(correlation_matrix,annot=True)


# In[39]:


###lets build for few variables
#print(train.shape) ####(1460, 81)
#print(correlation_matrix.shape)###(38, 38) 
#print(correlation_matrix.columns)
#print(correlation_matrix)

#k = 10
print(correlation_matrix.nlargest(k,'SalePrice')['SalePrice'].index)
#print(correlation_matrix.nlargest(k,'SalePrice')['SalePrice'].index)   ##10 * 38


# In[44]:


cols = correlation_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.heatmap(cm,annot = True,yticklabels=cols.values, xticklabels=cols.values)


# In[45]:


#'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
#'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables
#'TotalBsmtSF' and '1stFloor' also seem to be twin brothers,but we will keep 'TotalBsmtSF'
#TotRmsAbvGrd' and 'GrLivArea', twin brothers again. 

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols])
plt.show()


# In[47]:


###'TotalBsmtSF' and 'GrLiveArea' are related to each other linearly shoes that the licear and the
##basement are dependend on each other.

###saleprice and the year built are mostly correlated and draws the exponential curve shows
##with the increase in the salesprice with the change of year


##MISSING DATA

##now check the missing data
#(a) whether it shows some pattern
#(b) Importance in the missing data

total = train.isnull().sum().sort_values(ascending  = False)
percentage = (train.isnull().sum()/train.isnull().count()).sort_values(ascending  = False)
combine_data = pd.concat([total,percentage],axis = 1,keys = ['total','percentage'])
print(combine_data.head(20))


# In[59]:


missing_value = (combine_data[combine_data['total']>1]).index
print(missing_value)  ###index where the missing_value are more than 1


# In[70]:


print(train.shape)
#print(train.columns)


# In[71]:


print(train['Electrical'].unique())


# In[72]:


print(train.loc[train['Electrical'].isnull()].index)


# In[76]:


train = train.drop(train.loc[train['Electrical'].isnull()].index)


# In[77]:


##verify the null values

print(train.isnull().sum().max())


# In[78]:


###Its time to remove outliers

## we will be doing analysis based on the standard deviation and will check using standardization
from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[84]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[97]:


##Bivariate Analysis

plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('Saleprice')
plt.title('Relationship between Living area and the sales price')
plt.show()


# In[95]:


###In the above there are two outliers whose living area is more but the price is less
##so these are outliers and we need to remove that outliers
outliers = train.sort_values(by = 'GrLivArea',ascending = False)[:2]###to get first two rows
#print(outliers)
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)


# In[98]:


##Now look at
plt.scatter(train['TotalBsmtSF'],train['SalePrice'])
plt.show()

###Here we dont find outliers which sould be deleted


# In[108]:


###plot probability plot
from scipy.stats import norm
sns.distplot(train['SalePrice'],fit = norm)


# In[110]:


from scipy import stats
res = stats.probplot(train['SalePrice'],plot = plt)


# In[111]:


##plotting the probabilty plot shows that the the sales price does not follow normal distribution
##and we can change by using log values
train['SalePrice'] = np.log(train['SalePrice'])


# In[113]:


sns.distplot(train['SalePrice'],fit = norm)


# In[114]:


stats.probplot(train['SalePrice'],plot = plt)


# In[115]:


###Now we can see that it is in line

##now we look for the another variable

sns.distplot(train['GrLivArea'],fit = norm)


# In[116]:


from scipy import stats
stats.probplot(train['GrLivArea'],plot = plt)


# In[117]:


train['GrLivArea'] = np.log(train['GrLivArea'])


# In[118]:


sns.distplot(train['GrLivArea'],fit = norm)


# In[119]:


from scipy import stats
stats.probplot(train['GrLivArea'],plot  = plt)


# In[120]:


sns.distplot(train['TotalBsmtSF'],fit  = norm)


# In[121]:


##here there are totalBstSF = 0 we are unable to apply log transformation here

from scipy import stats
stats.probplot(train['TotalBsmtSF'],plot = plt)


# In[124]:


###Ohh we can see so much change as compared to previous grap
plt.scatter(train['SalePrice'],train['GrLivArea'])
plt.show()


# In[126]:


## we didnt do anything for the TotalBsmtSF

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
#histogram and normal probability plot
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# In[127]:


plt.scatter(train['TotalBsmtSF'],train['SalePrice'])


# In[ ]:


##how much change here as compared to the previous data

