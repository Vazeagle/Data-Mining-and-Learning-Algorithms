import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('winequality-red.csv', delimiter=',')
X=df.drop('quality', axis=1)#x= observed data
print(df.head())
print(df.shape)
print(df.describe())
print(df.quality.value_counts())
print(X)
y=df.quality#y=labels / target to prediction
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=45)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

#SVM model
classification_model = svm.SVC(kernel='poly',degree=5,gamma='scale',coef0=5.2,cache_size=500)   #train model using taining sets
classification_model.fit(X_train,y_train)
y_prediction=classification_model.predict(X_test)#predict Y
print("Accuracy:",metrics.accuracy_score(y_test,y_prediction ))

# Model Precision: what percentage of positive tuples are labeled as such?
#print("Precision:",metrics.precision_score(y_test, y_prediction))
# Model Recall: what percentage of positive tuples are labelled as such?
#print("Recall:",metrics.recall_score(y_test, y_prediction))

