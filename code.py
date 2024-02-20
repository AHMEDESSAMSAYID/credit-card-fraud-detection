import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report





# import data files

data= pd.read_csv("D:\yk project\Banksim.csv")

# data preprocessing

print(data["age"].unique())
print(data["category"].unique())
print(data["merchant"].unique())




labelcode=preprocessing.LabelEncoder()

data["age"]=labelcode.fit_transform(data["age"])
data["category"]=labelcode.fit_transform(data["category"])
data["merchant"]=labelcode.fit_transform(data["merchant"])
data["gender"]=labelcode.fit_transform(data["gender"])
#data.plot.kde()

data
x=data.iloc[:,0:5]
y=data.iloc[:,5]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10)

sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)




classfier= GaussianNB()

classfier.fit(x_train,y_train)

y_pred=classfier.predict(x_test) 

cm =confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


print(classification_report(y_test, y_pred))
print(cm)


#SVM Algorıthim

from sklearn.svm import SVC
%matplotlib notebook

fig,ax1=plt.subplots()

df0= data[data["fraud"]==0]
df1=data[data["fraud"]==1]
fig,ax1=plt.subplots()
ax1.set_xlabel("category")
ax1.set_ylabel=("age")
ax1.scatter(df0["category"][:200],df0["age"][:200],color="blue")
ax2=ax1.twinx()
ax2.scatter(df1["category"][:200],df1["age"][:200],color="red")

 
plt.show()
clf=SVC()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
cm_svc =confusion_matrix(y_test, y_predict)
accuracy_score(y_test, y_predict)

cm_svc
