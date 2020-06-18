# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:29:54 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split

iris=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\KNN\\iris.csv")
help(train_test_split)

train,test=train_test_split(iris,test_size=0.2)
#for 3 neighbors
neigh=KNC(n_neighbors=3)
#Fitting with training data
neigh.fit(train.iloc[:,0:4],train.iloc[:,4])

#to find train accuracy

train_acc=np.mean(neigh.predict(train.iloc[:,0:4])==train.iloc[:,4])
#to find train accuracy

test_acc=np.mean(neigh.predict(test.iloc[:,0:4])==test.iloc[:,4])


#for 5 neighbors

neigh1=KNC(n_neighbors=5) 
neigh1.fit(train.iloc[:,0:4],train.iloc[:,4])

#To find train accuracy

train_acc=np.mean(neigh1.predict(train.iloc[:,0:4])==train.iloc[:,4])

#To find test accuracy

test_accc=np.mean(neigh1.predict(test.iloc[:,0:4])==test.iloc[:,4])


# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 

acc=[]
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:4],train.iloc[:,4])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:4])==train.iloc[:,4])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:4])==test.iloc[:,4])
    acc.append([train_acc,test_acc])





import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


plt.legend(["train","test"])


##############################Type2#############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC

iris=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\KNN\\iris.csv")

X=iris.iloc[:,0:4]
y=iris.iloc[:,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import scale

X_train=scale(X_train)
X_test=scale(X_test)

model=KNC(n_neighbors=5)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix , classification_report

print(confusion_matrix(y_test,y_pred))
pd.crosstab(y_test.values.flatten(),y_pred)

print(classification_report(y_test,y_pred))
































































































