#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("C:/Users/Ravi/Downloads/MLR/Datasets_MLR/Avacado_Price.csv")


# In[3]:


df.info()


# In[4]:


duplicated=df.duplicated()


# In[5]:


sum(duplicated)


# In[6]:


df.isna().sum()


# In[7]:


df.isna().sum()


# In[8]:


df.shape


# In[9]:


df.head()


# In[10]:



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[11]:


df.type=le.fit_transform(df.type)


# In[12]:


df.region=le.fit_transform(df.region)


# In[13]:


df.describe()


# In[14]:


plt.bar(height=df.Total_Volume,x=np.arange(1,18250,1))


# In[15]:


plt.hist(df.Total_Volume)


# In[16]:


plt.boxplot(df.Total_Volume)


# In[17]:




IQR=df.Total_Volume.quantile(0.75)-df.Total_Volume.quantile(0.25)
lower_limit=df.Total_Volume.quantile(0.25)-(IQR*1.5)
upper_limit=df.Total_Volume.quantile(0.75)-(IQR*1.5)


# In[18]:



outliers=np.where(df.Total_Volume>upper_limit,True,np.where(df.Total_Volume<lower_limit,True,False))
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['Total_Volume'])


# In[19]:


df_t=winsor.fit_transform(df[["Total_Volume"]])


# In[20]:


plt.boxplot(df_t)


# In[21]:


df.Total_Volume=df_t


# In[22]:


plt.boxplot(df.Total_Volume)


# In[23]:


plt.hist(df.Total_Volume)


# In[24]:


sns.jointplot(x=df.Total_Volume,y=df.AveragePrice)


# In[25]:


plt.bar(height=df.tot_ava1,x=np.arange(1,18250,1))


# In[26]:


plt.boxplot(df.tot_ava1)


# In[27]:




IQR=df.tot_ava1.quantile(0.75)-df.tot_ava1.quantile(0.25)
lower_limit=df.tot_ava1.quantile(0.25)-(IQR*1.5)
upper_limit=df.tot_ava1.quantile(0.75)-(IQR*1.5)


outliers=np.where(df.tot_ava1>upper_limit,True,np.where(df.tot_ava1<lower_limit,True,False))
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['tot_ava1'])


# In[28]:


df_t=winsor.fit_transform(df[["tot_ava1"]])


# In[29]:


df.tot_ava1=df_t


# In[30]:


plt.boxplot(df.tot_ava1)


# In[31]:


plt.bar(height=df.tot_ava1,x=np.arange(1,18250,1))


# In[32]:


plt.hist(df.tot_ava1)


# In[33]:


sns.jointplot(x=df.tot_ava1,y=df.AveragePrice)


# In[34]:


plt.boxplot(df.tot_ava2)


# In[35]:




IQR=df.tot_ava2.quantile(0.75)-df.tot_ava2.quantile(0.25)
lower_limit=df.tot_ava2.quantile(0.25)-(IQR*1.5)
upper_limit=df.tot_ava2.quantile(0.75)-(IQR*1.5)


outliers=np.where(df.tot_ava2>upper_limit,True,np.where(df.tot_ava2<lower_limit,True,False))
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['tot_ava2'])


# In[36]:


df_t=winsor.fit_transform(df[["tot_ava2"]])


# In[37]:


df.tot_ava2=df_t


# In[38]:


plt.boxplot(df.tot_ava2)


# In[39]:


plt.bar(height=df.tot_ava2,x=np.arange(1,18250,1))


# In[40]:


plt.hist(df.tot_ava2)


# In[41]:


sns.jointplot(x=df.tot_ava2,y=df.AveragePrice)


# In[42]:


plt.boxplot(df.tot_ava3)


# In[43]:




IQR=df.tot_ava3.quantile(0.75)-df.tot_ava3.quantile(0.25)
lower_limit=df.tot_ava3.quantile(0.25)-(IQR*1.5)
upper_limit=df.tot_ava3.quantile(0.75)-(IQR*1.5)


outliers=np.where(df.tot_ava3>upper_limit,True,np.where(df.tot_ava3<lower_limit,True,False))
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['tot_ava3'])


# In[44]:


df_t=winsor.fit_transform(df[["tot_ava3"]])


# In[45]:


df.tot_ava3=df_t


# In[46]:


plt.boxplot(df.tot_ava3)


# In[47]:


plt.bar(height=df.tot_ava3,x=np.arange(1,18250,1))


# In[48]:


plt.hist(df.tot_ava3)


# In[49]:


plt.boxplot(df.Total_Bags)


# In[50]:




IQR=df.Total_Bags.quantile(0.75)-df.Total_Bags.quantile(0.25)
lower_limit=df.Total_Bags.quantile(0.25)-(IQR*1.5)
upper_limit=df.Total_Bags.quantile(0.75)-(IQR*1.5)


outliers=np.where(df.Total_Bags>upper_limit,True,np.where(df.Total_Bags<lower_limit,True,False))
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['Total_Bags'])


# In[51]:


df_t=winsor.fit_transform(df[["Total_Bags"]])


# In[52]:


df.Total_Bags=df_t


# In[53]:


plt.boxplot(df.Total_Bags)


# In[54]:


plt.bar(height=df.Total_Bags,x=np.arange(1,18250,1))


# In[55]:


plt.hist(df.Total_Bags)


# In[56]:


sns.jointplot(x=df.Total_Bags,y=df.AveragePrice)


# In[57]:


plt.boxplot(df.Small_Bags)


# In[58]:




IQR=df.Small_Bags.quantile(0.75)-df.Small_Bags.quantile(0.25)
lower_limit=df.Small_Bags.quantile(0.25)-(IQR*1.5)
upper_limit=df.Small_Bags.quantile(0.75)-(IQR*1.5)


outliers=np.where(df.Small_Bags>upper_limit,True,np.where(df.Small_Bags<lower_limit,True,False))
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['Small_Bags'])


# In[59]:


df_t=winsor.fit_transform(df[["Small_Bags"]])


# In[60]:


df.Small_Bags=df_t


# In[61]:


plt.boxplot(df.Small_Bags)


# In[62]:


plt.bar(height=df.Small_Bags,x=np.arange(1,18250,1))


# In[63]:


plt.hist(df.Small_Bags)


# In[64]:


sns.jointplot(x=df.Small_Bags,y=df.AveragePrice)


# In[65]:


plt.boxplot(df.Large_Bags)


# In[66]:




IQR=df.Large_Bags.quantile(0.75)-df.Large_Bags.quantile(0.25)
lower_limit=df.Large_Bags.quantile(0.25)-(IQR*1.5)
upper_limit=df.Large_Bags.quantile(0.75)-(IQR*1.5)


outliers=np.where(df.Large_Bags>upper_limit,True,np.where(df.Large_Bags<lower_limit,True,False))
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['Large_Bags'])


# In[67]:


df_t=winsor.fit_transform(df[["Large_Bags"]])


# In[68]:


df.Large_Bags=df_t


# In[69]:


plt.boxplot(df.Large_Bags)


# In[70]:


plt.bar(height=df.Large_Bags,x=np.arange(1,18250,1))


# In[71]:


plt.hist(df.Large_Bags)


# In[72]:


plt.boxplot(df.Xlarge_Bags)


# In[73]:




IQR=df.Large_Bags.quantile(0.75)-df.Xlarge_Bags.quantile(0.25)
lower_limit=df.Xlarge_Bags.quantile(0.25)-(IQR*1.5)
upper_limit=df.Xlarge_Bags.quantile(0.75)-(IQR*1.5)


outliers=np.where(df.Xlarge_Bags>upper_limit,True,np.where(df.Xlarge_Bags<lower_limit,True,False))
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['Xlarge_Bags'])


# In[74]:


df_t=winsor.fit_transform(df[["Xlarge_Bags"]])


# In[75]:


df.Xlarge_Bags=df_t


# In[76]:


plt.boxplot(df.Xlarge_Bags)


# In[ ]:





# In[86]:


from scipy import stats
import pylab
stats.probplot(df.AveragePrice, dist = "norm", plot = pylab)
plt.show()


# In[79]:


sns.pairplot(df.iloc[:, :])


# In[80]:


df.corr()


# In[81]:


import statsmodels.formula.api as smf


# In[82]:


model1=smf.ols("AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3  + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region ",data=df).fit()


# In[83]:


model1.summary()


# In[84]:


import statsmodels.api as sm


# In[85]:


sm.graphics.influence_plot(model1)


# In[ ]:





# In[87]:


res_vol=smf.ols("Total_Volume ~  tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region ",data=df).fit().rsquared


# In[88]:


res_vol = 1/(1 - res_vol) 


# In[89]:


res_vol


# In[90]:


res_ava1=smf.ols("tot_ava1 ~  Total_Volume + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region ",data=df).fit().rsquared
res_ava1 = 1/(1 - res_ava1) 


# In[91]:


res_ava1


# In[92]:


res_ava2=smf.ols("tot_ava2 ~  Total_Volume + tot_ava1 + tot_ava3 + Total_Bags + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region ",data=df).fit().rsquared
res_ava2 = 1/(1 - res_ava2) 


# In[93]:


res_ava2


# In[94]:


res_ava3=smf.ols("tot_ava3 ~  Total_Volume + tot_ava1 + tot_ava2 + Total_Bags + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region ",data=df).fit().rsquared
res_ava3 = 1/(1 - res_ava3)


# In[95]:


res_ava3


# In[96]:


res_Total_Bags=smf.ols("Total_Bags ~  Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region ",data=df).fit().rsquared
res_Total_Bags = 1/(1 - res_Total_Bags)


# In[97]:


res_Total_Bags


# In[101]:


res_Small_Bags=smf.ols("Total_Bags ~  Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags +Large_Bags+ Xlarge_Bags+type+year+region ",data=df).fit().rsquared
res_Small_Bags = 1/(1 - res_Small_Bags)


# In[102]:


res_Small_Bags


# In[103]:


res_Large_Bags=smf.ols("Large_Bags ~  Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags +Small_Bags+ Xlarge_Bags+type+year+region ",data=df).fit().rsquared
res_Large_Bags = 1/(1 - res_Large_Bags)


# In[104]:


res_Large_Bags


# In[105]:


res_Xlarge_Bags=smf.ols("Xlarge_Bags ~  Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags +Small_Bags+ Large_Bags+type+year+region ",data=df).fit().rsquared
res_Xlarge_Bags = 1/(1 - res_Xlarge_Bags)


# In[106]:


res_Xlarge_Bags


# In[107]:


res_type=smf.ols("type ~  Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags +Small_Bags+ Large_Bags+Xlarge_Bags+year+region ",data=df).fit().rsquared
res_type = 1/(1 - res_type)


# In[108]:


res_type


# In[109]:


res_year=smf.ols("year ~  Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags +Small_Bags+ Large_Bags+Xlarge_Bags+type+region ",data=df).fit().rsquared
res_year = 1/(1 - res_year)


# In[110]:


res_year


# In[111]:


res_region=smf.ols("region ~  Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags +Small_Bags+ Large_Bags+Xlarge_Bags+type+year ",data=df).fit().rsquared
res_region = 1/(1 - res_region)


# In[112]:


res_region


# In[113]:



df1 = {'Variables':['region','Total_Volume', 'tot_ava1', 'tot_ava2','tot_ava3','Total_Bags','Small_Bags','Large_Bags','Xlarge_Bags','type','year'], 'VIF':[res_region,res_vol, res_ava1, res_ava2,res_ava3,res_Total_Bags,res_Small_Bags,res_Large_Bags,res_Xlarge_Bags,res_type,res_year]}
Vif_frame = pd.DataFrame(df1)  


# In[114]:


Vif_frame


# In[115]:


model2=smf.ols("AveragePrice ~   tot_ava1  + tot_ava3 + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region ",data=df).fit()


# In[116]:


model2.summary()


# In[123]:


pre1=model2.predict(df)


# In[127]:



res1 = df.AveragePrice - pre1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


# In[130]:


model3=smf.ols("AveragePrice ~   np.log(tot_ava1  + tot_ava3 + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region) ",data=df).fit()


# In[131]:


model3.summary()


# In[133]:


pre2=model3.predict(df)

res2 = df.AveragePrice - pre2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# In[134]:


model4=smf.ols("np.log(AveragePrice) ~   (tot_ava1  + tot_ava3 + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region) ",data=df).fit()


# In[135]:


model4.summary()


# In[ ]:





# In[136]:


pre3=model4.predict(df)

res3 = df.AveragePrice - pre3
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# In[137]:


data = {"MODEL":pd.Series(["model2", "Log model", "Exp model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3])}
table_rmse = pd.DataFrame(data)
table_rmse


# In[138]:


final_model=smf.ols("(AveragePrice) ~   (tot_ava1  + tot_ava3 + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region) ",data=df).fit()


# In[139]:


final_model.summary()


# In[141]:


pre_final=final_model.predict(df)


# In[142]:


pre_final


# In[143]:



res = final_model.resid
sm.qqplot(res)
plt.show()


# In[144]:


pre_final


# In[145]:



# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()


# In[146]:


sns.residplot(x = pred, y = df.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()


# In[122]:


sm.graphics.influence_plot(final_model)


# In[160]:



from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 0.2,random_state=0) # 20% test data


# In[161]:


model_train = smf.ols("AveragePrice ~  tot_ava1 +   tot_ava3 + Total_Bags + Small_Bags +Large_Bags+ Xlarge_Bags+type+year+region", data = df_train).fit()


# In[162]:


test_pred = model_train.predict(df_test)


# In[163]:



# test residual values 
test_resid = test_pred - df_test.AveragePrice


# In[164]:



# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# In[165]:


# train_data prediction
train_pred = model_train.predict(df_train)


# In[166]:



# train residual values 
train_resid  = train_pred - df_train.AveragePrice


# In[167]:



# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


# In[168]:


train_rmse


# In[ ]:





# In[ ]:





# In[ ]:




