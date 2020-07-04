import pandas as pd
import re
import random
from decimal import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

df = pd.read_csv('winequality-red.csv', delimiter=',')
X = df.drop('quality', axis=1)  # x= observed data
print(df.head())
print(df.shape)
print(df.describe())
print(df.quality.value_counts())
print(X)
y = df.quality  # y=labels / target to prediction
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test)
#print(X_test.head())
#print(X_test.shape)

# SVM model
classification_model = svm.SVC(kernel='poly', degree=5, gamma='scale', coef0=5.2, class_weight=None,cache_size=500) #train model using taining sets
classification_model.fit(X_train, y_train)
y_prediction = classification_model.predict(X_test)  # predict Y
accuracy = metrics.accuracy_score(y_test, y_prediction)
print("Accuracy:", accuracy)

# Model Precision: what percentage of positive tuples are labeled as such?
precision = metrics.precision_score(y_test, y_prediction, average='weighted', zero_division=0)  # zero_division='warn'
print("Precision:", precision)

# Model Recall: what percentage of positive tuples are labelled as such?
recall = metrics.recall_score(y_test, y_prediction, average='weighted', zero_division=0)  # zero_division='warn'
print("Recall:", recall)

f1_score = metrics.f1_score(y_test, y_prediction, average='weighted', zero_division=0)  # zero_division='warn'

print("F1 score (metricslib):", f1_score)

# ERWTHMA B--------------------------------------------------------------------------------------------------------------------------------------

remove_counter = 0


# randomly remove 33% of ph values of list
def remove33pH(input_dataframe):
    global remove_counter
    input_list1 = input_dataframe.values.tolist()
    print("\nConverted list of X_train\n", input_list1)
    rows_counter = len(input_list1)
    remove_counter = round((rows_counter * 33) / 100)  # round number
    rows_remove_pH = random.sample(range(rows_counter),remove_counter)  # return a list of remove_counter numbers from range 0 to rows_counter to know which random rows to remove
    i = 0
    while i < remove_counter:
        cur_row = rows_remove_pH[i]  # random row to not be sequential
        # ph exists in 9th column
        input_list1[cur_row].pop(8)
        input_list1[cur_row].insert(8, 'zero')  # anti zero None aka null in python xwris''
        i += 1
    print("\nConverted list of X_train -33%pH\n", input_list1)
    output_dataframe = pd.DataFrame(input_list1,
                                    columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                             'chlorides',
                                             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                                             'sulphates',
                                             'alcohol'])
    print("\nConverted dataframe of X_train -33%pH\n", output_dataframe)


    return [input_list1, output_dataframe]


# checks if removal works as it should
def check_work(some_list):
    none_counter = 0
    j = 0
    while j < len(some_list):
        for elem in some_list[j]:
            if elem == 'zero':  # anti zero None aka null in python xwris''
                none_counter += 1
        j += 1
    print("\nsize=", len(some_list))
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
    input_list = init_list.copy()
    rows_with_no_ph = []
    i = 0
    getcontext().prec = 6  # PRECISION OF  4 DECIMAL POINTS    #not needed using round
    sum = Decimal(0)
    while i < len(input_list):
        elem = input_list[i][8]  # for each element in the inside list aka each row
        if elem != 'zero':  # anti zero None aka null in python xwris''
            convDec = Decimal(elem)
            sum = sum + (convDec)
        else:
            rows_with_no_ph.append(i)  # save which rows have no ph value
        i += 1

    avg = sum / len(input_list)
    print("\nΜέσος Όρος στοιχείων pH= ", avg)
    k = 0
    print("\namount of rows with zero ph:", len(rows_with_no_ph))
    while k < len(rows_with_no_ph):  # for each element in the inside list aka each row
        print("\ncur row:", rows_with_no_ph[k], " k=", k)
        input_list[rows_with_no_ph[k]].pop(8)
        input_list[rows_with_no_ph[k]].insert(8, avg)
        k += 1
    new_list = input_list.copy()
    print("\nX_train_list with avg replace\n", new_list)
    new_dataframe = pd.DataFrame(new_list,
                                 columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                          'pH', 'sulphates', 'alcohol'])
    print("\nX_train_ with avg replace in pH null:\n", new_dataframe)

    return [new_list, new_dataframe]


# B.3 fill None with Logistic regression of 67%  of Xtraint that has ph(use it as train model)
def b3(init_list2):
    input_list2 = init_list2.copy()
    input_list_copy = []
    for item in init_list2:
        input_list_copy.append(item)


    rows_with_no_ph = [] #λιστα που περιέχει ποιές γραμμές ανοικουν στο 33% με σβησμένες τιμές
    X_train_split_67 = []   #λίστα η οποία έχει γραμμές του dataframe που εμειναν αναλοιωτες
    X_test_split_33 = []   #λιστα που περιέχει το 33% των τιμών που εφαγαν delete στο pH περιέχει zero ή None
    split_67_int = []   #to logistic regression 8elei integers san input α χρησιμοποιηθει ως νεο X_train για το LOGISTIC REGRESSION αφου βγάλω το h ως Y train
    split_33_int = []   #to logistic regression 8elei integers san input θα χρησιμοποιηθει ως νεο X_test για το LOGISTIC REGRESSION
    temp_sub_list67 = []  # προσωρινη lista gia eswterika integers stis listes
    temp_sub_list33 = []  # προσωρινη lista gia eswterika integers stis listes
    i = 0
    #print(init_list2)

    while i < len(init_list2):
        elem = input_list2[i][8]  # for each element in the inside list aka each row
        if elem != 'zero':  #αν ανήκει στο 67% χωρις διεγραμένο pH
            X_train_split_67.append(input_list2[i].copy())

            if len(temp_sub_list67)>0:
                temp_sub_list67.clear()# αδειασμα λιστας για νεα στοιχεία
            idx_check=0
            for sub_element in input_list2[i]:
                if idx_check == 0:
                    temp_sub_list67.append(int(
                        sub_element * 10))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 1:
                    temp_sub_list67.append(int(
                        sub_element * 1000))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 2:
                    temp_sub_list67.append(int(
                        sub_element * 100))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 3:
                    temp_sub_list67.append(int(
                        sub_element * 10))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 4:
                    temp_sub_list67.append(int(
                        sub_element * 1000))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 5:
                    temp_sub_list67.append(int(
                        sub_element))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 6:
                    temp_sub_list67.append(int(
                        sub_element))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 7:
                    temp_sub_list67.append(int(
                        sub_element * 100000))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 8:
                    temp_sub_list67.append(int(
                        sub_element * 100))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 9:
                    temp_sub_list67.append(int(
                        sub_element * 100))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 10:
                    temp_sub_list67.append(int(
                        sub_element * 10))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                idx_check+=1
            split_67_int.append(temp_sub_list67.copy())  # προσθηκη στη λίστα ως integer
            #print("\ntemplist67=", temp_sub_list67)
            #print("\nlist67=", split_67_int)

        else:   # zero None aka null in python xwris''
            rows_with_no_ph.append(i)  # save which rows have no ph value
            X_test_split_33.append(input_list2[i].copy())

            input_list_copy[i].pop(8)  # delete 'zero' or None

            if len(temp_sub_list33)>0:
                temp_sub_list33.clear()# αδειασμα λιστας για νεα στοιχεία
            idx_check = 0
            for sub_element in input_list_copy[i]:# δημιουργώ την λίστα που θα αποτελέσει το X_test του logistic regression
                if idx_check == 0:
                    temp_sub_list33.append(int(
                        sub_element * 10))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 1:
                    temp_sub_list33.append(int(
                        sub_element * 1000))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 2:
                    temp_sub_list33.append(int(
                        sub_element * 100))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 3:
                    temp_sub_list33.append(int(
                        sub_element * 10))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 4:
                    temp_sub_list33.append(int(
                        sub_element * 1000))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 5:
                    temp_sub_list33.append(int(
                        sub_element))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 6:
                    temp_sub_list33.append(int(
                        sub_element))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 7:
                    temp_sub_list33.append(int(
                        sub_element * 100000))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 8:
                    temp_sub_list33.append(int(
                        sub_element * 100))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 9:
                    temp_sub_list33.append(int(
                        sub_element * 100))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
                if idx_check == 10:
                    temp_sub_list33.append(int(
                        sub_element * 10))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe

                idx_check += 1
            split_33_int.append(temp_sub_list33.copy())  # προσθηκη στη λίστα ως integer
            #print("\ntemplist33=",temp_sub_list33)
            #print("\nlist33=", split_33_int)
        i += 1

    print("\namount of rows with zero ph:", len(rows_with_no_ph))
    print("\namount of rows with zero ph:",rows_with_no_ph)
    print("\nrows with zero ph deleted full:", split_33_int)
    print("\namount of rows with ph integer converted:", split_67_int)
    # ΝΕΑ DATAFRAME ΜΕ Χ και Υ ΓΙΑ  LOGISTIC REGRESSION -------------------------------
    X_Y_train_dataframe = pd.DataFrame(split_67_int,
                                     columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol'])
    print("\n New train dataframe with X and Y (Y=pH) for logistic regression:\n", X_Y_train_dataframe)

    X_log_reg = X_Y_train_dataframe.drop('pH', axis=1)  # x= observed data
    y_log_reg = X_Y_train_dataframe.pH  # y=labels / target to prediction
    print("\nNew X_train for logistic regression\n",X_log_reg)
    print("\nNew Y_train for logistic regression\n",y_log_reg,"\n\n")


    # ΝΕΑ TEST DATAFRAME ΓΙΑ  LOGISTIC REGRESSION
    X_test_dataframe = pd.DataFrame(split_33_int,
                                     columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','sulphates', 'alcohol'])
    print("\nNew X_test dataframe with  no pH:\n", X_test_dataframe)

    logreg = LogisticRegression(max_iter=1000000)  # αρχικοποιηση με default τιμες του logistic regression
    logreg.fit(X_log_reg, y_log_reg)
    y_prediction_pH = logreg.predict(X_test_dataframe)  # predict Y
    print("\n\n\n Y PREDICTION PH\n",y_prediction_pH)

    y_pred_list_pH = y_prediction_pH.tolist()


    print("\n list with pH predictions=\n",y_pred_list_pH)
    print("length of y test=",len(y_pred_list_pH))


    c2=0
    print("X_test_split_33",X_test_split_33)
    print("X_train_split_67", X_train_split_67)
    new_final_list = X_train_split_67.copy()
    row_count=0
    for it_elem in X_test_split_33:
        new_final_list.insert(rows_with_no_ph[row_count],it_elem)
        row_count+=1
    print("\nfinal list\n",new_final_list)
    for list_el in new_final_list:
        elem_of_list = list_el[8]  # for each element in the inside list aka each row
        if elem_of_list == 'zero':
            list_el.pop(8)
            list_el.insert(8,(y_pred_list_pH[c2]/100))
            c2+=1
    new_list_pH=new_final_list
    print("\nNum of pH changed = ",c2,"\n")
    print("\nSAKIIIS\n", new_list_pH)

    #ΝΕΑ ΛΙΣΤΑ ΜΕΤΑ ΑΠΟ ΕΠΕΞΕΡΓΑΣΙΑ ΚΑΙ ΣΥΜΠΛΉΡΩΣΗ ΤΩΝ ΚΕΝΏΝ ΜΕ  LOGISTIC REGRESSION
    new_dataframe_w_pH = pd.DataFrame(new_list_pH,
                                 columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                          'pH', 'sulphates', 'alcohol'])
    print("\nX_train_ with logistic regression in pH null:\n", new_dataframe_w_pH)

    return [new_list_pH, new_dataframe_w_pH]


def b4(init_list3):
    list_to_process = init_list3.copy()
    listX = init_list3.copy()

    #lista gia to X_train
    X_trainAll_w_pH = pd.DataFrame(listX,
                                      columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                               'pH', 'sulphates', 'alcohol'])
    #dataframe me mono X_train all
    X_trainAll_no_pH = X_trainAll_w_pH.drop('pH', axis=1)



    rows_with_zero_ph = []  # λιστα που περιέχει ποιές γραμμές ανοικουν στο 33% με σβησμένες τιμές
    X_train_67_w_pH = []  # λίστα η οποία έχει γραμμές του dataframe που εμειναν αναλοιωτες
    X_test_33_zero_pH = []  # λιστα που περιέχει το 33% των τιμών που εφαγαν delete στο pH περιέχει zero ή None
    X_test_33_no_pH = []  # λιστα που περιέχει το 33% των τιμών που εφαγαν delete στο pH περιέχει zero ή None
    i=0
    while i < len(list_to_process):
        elem = list_to_process[i][8]  # for each element in the inside list aka each row
        if elem != 'zero':  #αν ανήκει στο 67% χωρις διεγραμένο pH
            X_train_67_w_pH.append(list_to_process[i].copy())


        else:   # zero None aka null in python xwris''
            rows_with_zero_ph.append(i)  # save which rows have no ph value
            X_test_33_zero_pH.append(list_to_process[i].copy())
        i += 1

        temp_lista = X_test_33_zero_pH.copy()
        X_test_df_w_pH = pd.DataFrame(temp_lista,
                                          columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                                   'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                                                   'density',
                                                   'pH', 'sulphates', 'alcohol'])
        X_test_df_no_pH = X_test_df_w_pH.drop('pH', axis=1)

    print("\nX_test_33_zero_pH=\n", X_test_33_zero_pH)
    print("\ntemp_lista=\n", temp_lista)
    print("\nX_test_33_no_pH=\n",X_test_df_no_pH)

    X_kai_Y_train_67_w_pH = pd.DataFrame(X_train_67_w_pH,
                                          columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                                   'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                                                   'density',
                                                   'pH', 'sulphates', 'alcohol'])

    X_train_kmean = X_kai_Y_train_67_w_pH.drop('pH', axis=1)
    Y_train_67_pH = X_kai_Y_train_67_w_pH.pH

    #X_trainAll_no_pH
    #επιλεγουμε να χρησιμοποιησουμε 4 clusters για τα kmeans
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    centers=kmeans.fit(X_trainAll_no_pH).cluster_centers_

    #y_kmeans = kmeans.predict()
    print("\ncenters=",centers)


    return ["lisit", "dataframe"]

temp33 = remove33pH(X_train)  # call function to empty 33% random fromof ph
X_train_list33 = temp33[0]  # resulting list -33%
X_train33 = temp33[1]  # resulting dataframe -33%
b1_input = pd.DataFrame(X_train_list33,columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol'])
b2_input = X_train33.values.tolist()
b3_input = X_train33.values.tolist()
b4_input = X_train33.values.tolist()
print("67=",temp33[1])

check_work(X_train_list33)

# Ερώτημα b1
tempb1 = b1(b1_input)
X_train_listb1 = tempb1[0]  # list with removed pH element
X_trainb1 = tempb1[1]  # dataframe with removed pH column


# Ερώτημα b2
tempb2 = b2(b2_input)
X_train_listb2 = tempb2[0]  # list with M.O. at removed pH values
X_trainb2 = tempb2[1]  # dataframe with M.O. at removed pH values


# Ερώτημα b3
# Logistic Regression ειναι binary δηλαδή yes ή no

tempb3 = b3(b3_input)
X_train_listb3 = tempb3[0]  # list with logistic regression pH values
X_trainb3 = tempb3[1]  # dataframe with logistic regression  pH values



# Ερώτημα b4
# K-means
tempb4 = b4(b4_input)
X_train_listb4 = tempb4[0]  # list with K-means pH values
X_trainb4 = tempb4[1]  # dataframe with K-means  pH values


#############################################################################
#############################################################################
#############################################################################
######################## RESULTS FROM METRICS ###############################

classification_model.fit(X_trainb1, y_train)
X_test_del_pH = X_test.drop('pH', axis=1)
y_predictionb1 = classification_model.predict(X_test_del_pH)  # predict Y
accuracyb1 = metrics.accuracy_score(y_test, y_predictionb1)
print("\nAccuracy of b1:",accuracyb1)
f1_score = metrics.f1_score(y_test, y_predictionb1, average='weighted', zero_division=0)
print("\nf1 score_B1: ",f1_score)
precision = metrics.precision_score(y_test, y_predictionb1, average='weighted', zero_division=0)  # zero_division='warn'
print("Precision_B1: ", precision)
recall = metrics.recall_score(y_test, y_predictionb1, average='weighted', zero_division=0)  # zero_division='warn'
print("Recall_B1: ", recall)

classification_model.fit(X_trainb2, y_train)
y_predictionb2 = classification_model.predict(X_test)  # predict Y
accuracyb2 = metrics.accuracy_score(y_test, y_predictionb2)
print("\nAccuracy of b2:",accuracyb2)
f1_score = metrics.f1_score(y_test, y_predictionb2, average='weighted', zero_division=0)
print("\nf1 score_B2:",f1_score)
precision = metrics.precision_score(y_test, y_predictionb2, average='weighted', zero_division=0)  # zero_division='warn'
print("Precision_B2:", precision)
recall = metrics.recall_score(y_test, y_predictionb2, average='weighted', zero_division=0)  # zero_division='warn'
print("Recall_B2:", recall)

classification_model.fit(X_trainb3, y_train)
y_predictionb3 = classification_model.predict(X_test)  # predict Y
accuracyb3 = metrics.accuracy_score(y_test, y_predictionb3)
print("\nAccuracy of b3: ",accuracyb3)
f1_score = metrics.f1_score(y_test, y_predictionb3, average='weighted', zero_division=0)
print("\nf1 score_B3: ",f1_score)
precision = metrics.precision_score(y_test, y_predictionb3, average='weighted', zero_division=0)  # zero_division='warn'
print("Precision_B3: ", precision)
recall = metrics.recall_score(y_test, y_predictionb3, average='weighted', zero_division=0)  # zero_division='warn'
print("Recall_B3: ", recall)



#classification_model.fit(X_trainb4, y_train)
#y_predictionb4 = classification_model.predict(X_test)  # predict Y
#accuracyb4 = metrics.accuracy_score(y_test, y_prediction)
#print("Accuracy of b4:"accuracyb4)
#f1_score = metrics.f1_score(y_test, y_prediction, average='weighted', zero_division=0)
