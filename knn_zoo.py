# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:12:44 2020

@author: Rohith
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC

zoo=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\KNN\\Zoo.csv")

train,test=train_test_split(zoo,test_size=0.2)

neigh=KNC(n_neighbors=3)
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])

#To find train accuracy

train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])

#To find test accuracy

test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])

#Similarly for n_neighbors=7

neigh=KNC(n_neighbors=7)
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])

#To find train accuracy

train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])

#To find test accuracy

test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])

acc=[]

for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
    test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
    acc.append([train_acc,test_acc])
    
    
    
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")




#######################################Type 2###################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split 

zoo=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\KNN\\Zoo.csv")
zoo.dtypes

X=zoo.iloc[:,1:17]
y=zoo.iloc[:,17]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.preprocessing import scale

X_train=scale(X_train)
X_test=scale(X_test)


neigh=KNC(n_neighbors=7)
neigh.fit(X_train,y_train)

y_pred=neigh.predict(X_test)

from sklearn.metrics import confusion_matrix , classification_report

print(confusion_matrix(y_test,y_pred))
pd.crosstab(y_test.values.flatten(),y_pred)

print(classification_report(y_test,y_pred))


























plt.legend(["train","test"])
    





