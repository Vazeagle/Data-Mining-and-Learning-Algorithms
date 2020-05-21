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

#ERWTHMA2--------------------------------------------------------------------------------------------------------------------------------------
#ERWTHMA2--------------------------------------------------------------------------------------------------------------------------------------
#ERWTHMA2--------------------------------------------------------------------------------------------------------------------------------------
#ERWTHMA2--------------------------------------------------------------------------------------------------------------------------------------


#ph_to_remove=X_train.pH #get only ph from data trainset
#train_to_remove=ph_to_remove.sample(frac=0.33, random_state=45)# 0,33% to remove from training dataset
#X_train=X_train.drop(train_to_remove.index) λαθος γιατί κανει delete ολα τα rows όχι μονο το συγκεκριμενο της στήλης


remove_counter=0
#randomly remove 33% of ph values of list
def remove33pH(input_dataframe):
    global remove_counter
    input_list = input_dataframe.values.tolist()
    print("\nConverted list of X_train\n",input_list)
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
    print("\nConverted list of X_train -33%pH\n", input_list)
    output_dataframe = pd.DataFrame(input_list,
                           columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                    'alcohol'])
    print("\nConverted dataframe of X_train -33%pH\n", output_dataframe)

    return [input_list,output_dataframe]

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
    print("None expected base on 33%: ", remove_counter)


# B.1 Remove Column from dataframe
def b1(init_dataframe):
    new_dataframe = init_dataframe.drop('pH', axis=1)
    new_list = new_dataframe.values.tolist()
    print("X_train_list with with removed pH element:\n", new_list)
    print("X_train with removed pH column:\n", new_dataframe)
    return [new_list, new_dataframe]  # return new list and dataframe


# B.2 fill None with M.O. of column lists
def b2(init_list):
    input_list=init_list
    rows_with_no_ph=[]
    i = 0
    sum = 0
    while i < len(input_list):
        elem = input_list[i][8]  # for each element in the inside list aka each row
        if elem != 'zero':  # anti zero None aka null in python xwris''
            sum = sum + float(elem)
        else:
            rows_with_no_ph.append(i) #save which rows have no ph value
        i += 1
    avg = sum / len(input_list)
    print("\nΜέσος Όρος στοιχείων pH= ", avg)
    k = 0
    print("\namount of rows with zero ph:",len(rows_with_no_ph))
    while k < len(rows_with_no_ph): # for each element in the inside list aka each row
        print("\ncur row:",rows_with_no_ph[k]," k=",k)
        input_list[rows_with_no_ph[k]].pop(8)
        input_list[rows_with_no_ph[k]].insert(8, avg)
        k += 1
    new_list = input_list
    print("\nX_train_list with avg replace\n", new_list)
    new_dataframe = pd.DataFrame(new_list,
                                     columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                              'pH', 'sulphates', 'alcohol'])
    print("\nX_train_ with avg replace in pH null:\n",new_dataframe)

    return [new_list, new_dataframe]


temp33 = remove33pH(X_train)
X_train_list33 = temp33[0]
X_train33 = temp33[1]

check_work(X_train_list33)


#Ερώτημα b1
tempb1 = b1(X_train33)
X_train_listb1 = tempb1[0]   #list that with removed pH element
X_trainb1 = tempb1[1]    #dataframe that with removed pH column

#Ερώτημα b2
tempb2 = b2(X_train_list33)
X_train_listb2 = tempb2[0]
X_trainb2 = tempb2[1]

#To Do Ερώτημα β3 β4
#ΑΛΛΑΓΗ ΣΕ PRECISION 4 ΔΕΚΑΔΙΚΩΝ ΨΗΦΙΩΝ ΣΤΟ FLOAT ΓΙΑ ΠΙΟ ΟΚ ΔΕΔΟΜΕΝΑ
