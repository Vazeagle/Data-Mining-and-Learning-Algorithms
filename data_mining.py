import pandas as pd
import re
import random
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
classification_model = svm.SVC(kernel='poly',degree=5,gamma='scale',coef0=5.2, class_weight=None, cache_size=500)   #train model using taining sets
classification_model.fit(X_train,y_train)
y_prediction=classification_model.predict(X_test)#predict Y
accuracy = metrics.accuracy_score(y_test,y_prediction )
print("Accuracy:",accuracy)

#Model Precision: what percentage of positive tuples are labeled as such?
precision = metrics.precision_score(y_test, y_prediction,average='weighted', zero_division=0)#zero_division='warn'
print("Precision:",precision)

# Model Recall: what percentage of positive tuples are labelled as such?
recall = metrics.recall_score(y_test, y_prediction,average='weighted', zero_division=0)#zero_division='warn'
print("Recall:",recall)


f1_score = metrics.f1_score(y_test, y_prediction, average='weighted', zero_division=0)#zero_division='warn'

print("F1 score (metricslib):",f1_score)

##########SOS ΕΡΩΤΗΣΗ ΤΙ AVERAGE ΘΕΛΟΥΜΕ?? binary micro macro weighted samples??????????

#ERWTHMA2
#ph_to_remove=X_train.pH #get only ph from data trainset
#train_to_remove=ph_to_remove.sample(frac=0.33, random_state=45)# 0,33% to remove from training dataset
#X_train=X_train.drop(train_to_remove.index) λαθος γιατί κανει delete ολα τα rows όχι μονο το συγκεκριμενο της στήλης

#randomly remove 33% of ph values of list
def remove33pH(input_list):
    rows_counter=len(input_list)
    remove_counter = round((rows_counter*33)/100)#round number
    rows_remove_pH = random.sample(range(rows_counter), remove_counter)#return a list of remove_counter numbers from range 0 to rows_counter
    i=0
    while i<remove_counter:
        cur_row=rows_remove_pH[i]#random row to not be sequential
        #ph exists in 9th column
        input_list[cur_row].pop(8)
        input_list[cur_row].insert(8,'zero')#anti zero None aka null in python xwris''
        i+=1
    return remove_counter

#checks if removal works as it should
def check_work(some_list):
    none_counter = 0
    j = 0
    while j < len(some_list):
        for elem in some_list[j]:
            if elem == 'zero':  #anti zero None aka null in python xwris''
                none_counter += 1
        j += 1
    print("\nsize=",len(some_list))
    print("None count: ", none_counter)
    print("None expected base on 33%: ", remove33pH(some_list))

X_train_list = X_train.values.tolist()
print("\nX_train:\n")
print(X_train_list)
print(X_train_list[0][8])
print(X_train_list[1][8])
remove33pH(X_train_list)
print(X_train_list)
print(X_train_list[0][8])
print(X_train_list[1][8])
check_work(X_train_list)

#list to dataframe conversion

#convert back to dataframe with zeros
X_train = pd.DataFrame(X_train_list,columns = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'])
print("\nX_train33% :\n")
print(X_train.head())
print(X_train.shape)
