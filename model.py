import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle


# Importing the dataset
data = pd.read_csv('iris.csv')




#Converting categorical to numerical so we can use knn
data['variety']=data['variety'].map({'Setosa':0,'Versicolor':1,'Virginica':2})

#Choosing the features as x and the target as y
x=data[['sepal.length','sepal.width','petal.length','petal.width']]
y=data['variety']


#KNN algorithm

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30) #split our data with test size of 20%

knn=KNeighborsClassifier(n_neighbors=20) #build our knn classifier
knn.fit(x_train,y_train) #Training KNN classifier
y_pred=knn.predict(x_test)  #Testing
print('Acuuracy=',accuracy_score(y_pred,y_test))
pickle.dump(knn,open('model.pkl','wb'))
