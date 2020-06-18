# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:51:43 2020

@author: Rohith
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split

glass=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\KNN\\glass.csv")

glass_scale=pd.DataFrame(scale(glass.iloc[:,0:9]))
glass_new=pd.concat([glass_scale,glass['Type']],axis=1)

train,test=train_test_split(glass_new,test_size=0.2)



neigh=KNC(n_neighbors=3)
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])

train_acc=np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
test_acc=np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])


acc=[]

for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc=np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc=np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    
    acc.append([train_acc,test_acc])
    
    
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


plt.legend(["train","test"])


################################Type2#############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split

glass=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\KNN\\glass.csv")

X=glass.iloc[:,0:9]
y=glass.iloc[:,9]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.preprocessing import scale

X_train=scale(X_train)
X_test=scale(X_test)

model=KNC(n_neighbors=3)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix , classification_report

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))































