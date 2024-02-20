#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Data loading,processing
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

#vizalation
import matplotlib.pyplot as plt
import seaborn as sns

#metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


#Models

from sklearn import preprocessing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


# In[10]:


data= pd.read_csv("D:\yk project\Banksim.csv")
data.head(5)


# In[11]:


data.info()


# In[12]:


df_non_fraud= data[data["fraud"]==0]
df_fraud=data[data["fraud"]==1]

sns.countplot(x="fraud",data=data)
plt.title("Dolandırıcılık Ödemelerinin Sayısı")
plt.show()
print("Normal örneklerin sayısı :",df_non_fraud.fraud.count())
print("Dolandırıcılık örneklerin sayısı :",df_fraud.fraud.count())


# In[13]:


# Plot histograms of the amounts in fraud and non-fraud data 
plt.figure(figsize=(30,10))
sns.boxplot(x=data.category,y=data.amount)
plt.title("Boxplot for the Amount spend in category")
plt.ylim(0,4000)
plt.legend()
plt.show()


# In[14]:


print("Mean feature",data.groupby("category")[["amount","fraud"]].mean())


# In[16]:


plt.hist(df_non_fraud.amount,alpha=0.5,label="dolandırıcı",bins=100)
plt.hist(df_fraud.amount,alpha=0.5,label="dolandırıcı degil",bins=100)
plt.title("dolandırıcı ve dolandırıcı olmayan ödemelerm")
plt.ylim(0,10000)
plt.xlim(0,1000)
plt.legend()
plt.show()


# In[17]:


# make the type of object columns categorical to simplify the conversion process
data_categorical= data.select_dtypes(include=['object']).columns
data_categorical

for col in data_categorical:
    data[col]= data[col].astype('category')

data[data_categorical]=data[data_categorical].apply(lambda x:x.cat.codes)
data.head(5)
#Define the independent variable (X) and the dependent/target variable y

x=data.drop(['fraud'],axis=1)
y=data['fraud']


# In[18]:


#Oversampling with SMOTE

sm=SMOTE(random_state=20)
X_res,y_res =sm.fit_resample(x,y)
y_res=pd.DataFrame(y_res)
print(y_res.value_counts())


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.3,random_state=42,shuffle=True,stratify=y_res)

# Describe the architecture of a feedforward neural network
input_dim = X_train.shape[1]
output_dim=1
hidden_units=[64,32]

# Creating a feedforward neural network model

model=tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(input_dim)))

for units in hidden_units:
    model.add(tf.keras.layers.Dense(units,activation='relu'))
model.add(tf.keras.layers.Dense(output_dim,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_train=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=512,epochs=10)


# In[20]:


classfier= GaussianNB()

classfier.fit(X_train,y_train)

y_pred=classfier.predict(X_test) 

cm =confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


print(classification_report(y_test, y_pred))
print(cm)


# In[21]:


knn=KNeighborsClassifier(n_neighbors=5,p=1)

knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print("Classification Report for K-Nearest Neigbours:\n",classification_report(y_test,y_pred))
print("Confusion Matrix of K-Nearest Neigbours: \n",confusion_matrix(y_test, y_pred))     


# In[22]:


XGboost=xgb.XGBClassifier(max_depth=6,learning_rate=0.05, n_estimators=400, 
                                objective="binary:hinge", booster='gbtree', 
                                n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                                subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                                scale_pos_weight=1, base_score=0.5, random_state=42)

XGboost.fit(X_train,y_train)

XGBoost_CLF = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, 
                                objective="binary:hinge", booster='gbtree', 
                                n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                                subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                                scale_pos_weight=1, base_score=0.5, random_state=42)

XGBoost_CLF.fit(X_train,y_train)

y_pred = XGBoost_CLF.predict(X_test)

print("Classification Report for XGBoost: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of XGBoost: \n", confusion_matrix(y_test,y_pred))


# In[ ]:





# ## 

# In[ ]:




