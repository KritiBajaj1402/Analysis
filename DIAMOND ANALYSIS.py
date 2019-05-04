#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir("C:/Users/130901/Desktop/Diamond")


# In[87]:


train_Data = pd.read_csv("diamonds.csv")


# In[88]:


print(train_Data.shape)


# In[89]:


train_Data.columns.values


# In[90]:


train_Data = train_Data.drop(['Unnamed: 0'],axis = 1)


# In[91]:


train_Data.columns.values


# In[92]:


plt.figure(figsize = (20,20))
sns.heatmap(train_Data.corr(),annot = True)
plt.savefig('Correlation matrix.png')


# In[93]:


sns.pairplot(train_Data)
plt.savefig('PairPlot.png')


# In[94]:


train_Data.describe()


# In[95]:


print((train_Data.x == 0).sum())


# In[96]:


print((train_Data.y == 0).sum())


# In[97]:


print((train_Data.z == 0).sum())


# In[98]:


train_Data[['x','y','z']] = train_Data[['x','y','z']].replace(0,np.NaN)


# In[99]:


train_Data.dropna(inplace = True)


# In[100]:


print(train_Data.shape)


# In[101]:


train_Data.isnull().sum()


# In[102]:


train_Data.hist(bins = 150)
plt.savefig('Histogram.png')


# In[103]:


train_Data.cut.nunique()


# In[104]:


sns.countplot(train_Data.cut)
plt.savefig('Cut - countplot.png')


# In[105]:


sns.countplot(train_Data.color)
plt.savefig('color - countplot.png')


# In[106]:


sns.countplot(train_Data.clarity)
plt.savefig('clarity - countplot.png')


# In[107]:


encoded = pd.get_dummies(train_Data)
print(encoded.shape)


# In[108]:


cols = encoded.columns


# In[109]:


print(cols)


# In[110]:


clean_data = pd.DataFrame(encoded,columns = cols)


# In[111]:


print(clean_data.shape)


# In[112]:


from sklearn.preprocessing import StandardScaler


# In[113]:


obj1 = StandardScaler()


# In[114]:


numericals = pd.DataFrame(obj1.fit_transform(clean_data[['carat', 'depth', 'table','x', 'y', 'z']]),columns = ['carat', 'depth', 'table','x', 'y', 'z'],index=clean_data.index)


# In[115]:


print(numericals.head(5))


# In[116]:


clean_data = clean_data.drop(['carat', 'depth', 'table','x', 'y', 'z'],axis = 1)


# In[117]:


print(clean_data.shape)


# In[118]:


combine_data = pd.concat([clean_data,numericals],axis = 1)


# In[119]:


print(combine_data.shape)


# In[120]:


combine_data.isnull().sum()


# In[121]:


Y = combine_data.price


# In[122]:


print(combine_data.shape)


# In[123]:


combine_data = combine_data.drop(['price'],axis = 1)


# In[124]:



combine_data = combine_data.drop(['cut_Fair','color_D','clarity_SI2'],axis = 1)


# In[125]:


from sklearn.model_selection import train_test_split


# In[126]:


train_x,test_x,train_y,test_y = train_test_split(combine_data,Y,test_size = 0.3,random_state = 0)


# In[127]:


from sklearn.linear_model import LinearRegression


# In[128]:


obj1 = LinearRegression()


# In[129]:


print(train_x.shape)


# In[130]:


obj1.fit(train_x,train_y)


# In[132]:


y_pred = obj1.predict(test_x)


# In[134]:


print(obj1.score(test_x,test_y))


# In[136]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(test_y,y_pred)


# In[138]:


from sklearn.metrics import mean_squared_error
mean_squared_error(test_y,y_pred)


# In[139]:


from sklearn.metrics import r2_score
r2_score(test_y,y_pred)


# In[140]:


print(test_x.shape)


# In[141]:


n = test_x.shape[0]
p = test_x.shape[1]


# In[143]:


from sklearn.linear_model import Lasso


# In[144]:


model2 = Lasso()


# In[145]:


model2.fit(train_x,train_y)


# In[146]:


y_pred = model2.predict(test_x)


# In[147]:


model2.score(test_x,test_y)


# In[148]:


mean_absolute_error(test_y,y_pred)


# In[149]:


mean_squared_error(test_y,y_pred)


# In[151]:


r2_score(test_y,y_pred)


# In[152]:


from sklearn.linear_model import Ridge


# In[153]:


model3 = Ridge()


# In[154]:


model3.fit(train_x,train_y)


# In[155]:


y_pred = model3.predict(test_x)


# In[156]:


model3.score(test_x,test_y)


# In[157]:


mean_squared_error(test_y,y_pred)


# In[158]:


mean_absolute_error(test_y,y_pred)


# In[159]:


r2_score(test_y,y_pred)


# In[161]:


import statsmodels.formula.api as sm


# In[164]:


combine_data = np.append(arr = np.ones((combine_data.shape[0],1)).astype(int),values = clean_data,axis = 1)


# In[170]:


print(combine_data.shape)


# In[171]:


np.arange(0,22)


# In[172]:


X_opt = combine_data[:,[0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21]]


# In[173]:


model2 = sm.OLS(endog = Y,exog = X_opt).fit()


# In[174]:


model2.summary()


# In[175]:


X_opt = combine_data[:,[0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,16,
       17, 18, 19, 20, 21]]


# In[176]:


model2 = sm.OLS(endog = Y,exog = X_opt).fit()


# In[177]:


model2.summary()


# In[ ]:




