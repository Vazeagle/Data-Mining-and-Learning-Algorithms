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
import matplotlib.pyplot as plt

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
print(X_test.head())
print(X_test.shape)

# SVM model
classification_model = svm.SVC(kernel='poly', degree=5, gamma='scale', coef0=5.2, class_weight=None,
                               cache_size=500)  # train model using taining sets
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

##########SOS ΕΡΩΤΗΣΗ ΤΙ AVERAGE ΘΕΛΟΥΜΕ?? binary micro macro weighted samples??????????

# ERWTHMA2--------------------------------------------------------------------------------------------------------------------------------------
# ERWTHMA2--------------------------------------------------------------------------------------------------------------------------------------
# ERWTHMA2--------------------------------------------------------------------------------------------------------------------------------------
# ERWTHMA2--------------------------------------------------------------------------------------------------------------------------------------


# ph_to_remove=X_train.pH #get only ph from data trainset
# train_to_remove=ph_to_remove.sample(frac=0.33, random_state=45)# 0,33% to remove from training dataset
# X_train=X_train.drop(train_to_remove.index) λαθος γιατί κανει delete ολα τα rows όχι μονο το συγκεκριμενο της στήλης


remove_counter = 0


# randomly remove 33% of ph values of list
def remove33pH(input_dataframe):
    global remove_counter
    input_list = input_dataframe.values.tolist()
    print("\nConverted list of X_train\n", input_list)
    rows_counter = len(input_list)
    remove_counter = round((rows_counter * 33) / 100)  # round number
    rows_remove_pH = random.sample(range(rows_counter),remove_counter)  # return a list of remove_counter numbers from range 0 to rows_counter to know which random rows to remove
    i = 0
    while i < remove_counter:
        cur_row = rows_remove_pH[i]  # random row to not be sequential
        # ph exists in 9th column
        input_list[cur_row].pop(8)
        input_list[cur_row].insert(8, 'zero')  # anti zero None aka null in python xwris''
        i += 1
    print("\nConverted list of X_train -33%pH\n", input_list)
    output_dataframe = pd.DataFrame(input_list,
                                    columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                             'chlorides',
                                             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                                             'sulphates',
                                             'alcohol'])
    print("\nConverted dataframe of X_train -33%pH\n", output_dataframe)

    return [input_list, output_dataframe]


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
    input_list = init_list
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
    new_list = input_list
    print("\nX_train_list with avg replace\n", new_list)
    new_dataframe = pd.DataFrame(new_list,
                                 columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                          'pH', 'sulphates', 'alcohol'])
    print("\nX_train_ with avg replace in pH null:\n", new_dataframe)

    return [new_list, new_dataframe]


# B.3 fill None with Logistic regression of 67%  of Xtraint that has ph(use it as train model)
def b3(init_list2):
    input_list = init_list2
    input_list_copy = input_list.copy()
    rows_with_no_ph = [] #λιστα που περιέχει ποιές γραμμές ανοικουν στο 33% με σβησμένες τιμές
    X_train_split_67 = []   #λίστα η οποία έχει γραμμές του dataframe που εμειναν αναλοιωτες
    X_test_split_33 = []   #λιστα που περιέχει το 33% των τιμών που εφαγαν delete στο pH περιέχει zero ή None
    split_67_int = []   #to logistic regression 8elei integers san input α χρησιμοποιηθει ως νεο X_train για το LOGISTIC REGRESSION αφου βγάλω το h ως Y train
    split_33_int = []   #to logistic regression 8elei integers san input θα χρησιμοποιηθει ως νεο X_test για το LOGISTIC REGRESSION
    temp_sub_list67 = []  # προσωρινη lista gia eswterika integers stis listes
    temp_sub_list33 = []  # προσωρινη lista gia eswterika integers stis listes
    i = 0
    print(init_list2)#----------------sososososososososososososososososososososososososoSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOSOS

    while i < len(init_list2):
        elem = input_list[i][8]  # for each element in the inside list aka each row
        if elem != 'zero':  # anti zero None aka null in python xwris''
            X_train_split_67.append(input_list[i])

            if len(temp_sub_list67)>0:
                temp_sub_list67.clear()# αδειασμα λιστας για νεα στοιχεία

            for sub_element in input_list[i]:
                temp_sub_list67.append(int(round(sub_element)))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
            split_67_int.append(temp_sub_list67)  # προσθηκη στη λίστα ως integer
            print("\ntemplist67=", temp_sub_list67)
            print("\nlist67=", split_67_int)

        else:   #αν ανήκει στο 67% χωρις διεγραμένο pH
            rows_with_no_ph.append(i)  # save which rows have no ph value
            X_test_split_33.append(input_list[i])

            input_list_copy[i].pop(8)  # delete 'zero' or None

            if len(temp_sub_list33)>0:
                temp_sub_list33.clear()# αδειασμα λιστας για νεα στοιχεία

            for sub_element in input_list_copy[i]:  # δημιουργώ την λίστα που θα αποτελέσει το X_test του logistic regression
                temp_sub_list33.append(int(round(sub_element)))  # εδω ειναι μια λιστα που περιέχει όλες τις τιμές σαν int μιας σειρας απο το dataframe
            split_33_int.append(temp_sub_list33)  # προσθηκη στη λίστα ως integer
            print("\ntemplist33=",temp_sub_list33)
            print("\nlist33=", split_33_int)
        i += 1

    print("\namount of rows with zero ph:", len(rows_with_no_ph))
    print("\nrows with zero ph deleted full:", split_33_int)
    print("\namount of rows with ph integer converted:", split_67_int)
    # ΝΕΑ DATAFRAME ΜΕ Χ και Υ ΓΙΑ  LOGISTIC REGRESSION -------------------------------
    X_Y_train_dataframe = pd.DataFrame(split_67_int,
                                     columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol'])
    print("\n New train dataframe with X and Y (Y=pH) for logistic regression:\n", X_Y_train_dataframe)

    X_log_reg = X_Y_train_dataframe.drop('pH', axis=1)  # x= observed data
    y_log_reg = X_Y_train_dataframe.pH  # y=labels / target to prediction
    print("\nNew X_train for logistic regression",X_log_reg)
    print("\nNew Y_train for logistic regression",y_log_reg,"\n\n")


    # ΝΕΑ TEST DATAFRAME ΓΙΑ  LOGISTIC REGRESSION
    X_test_dataframe = pd.DataFrame(split_33_int,
                                     columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','sulphates', 'alcohol'])
    print("\nNew X_test dataframe with  no pH:\n", X_test_dataframe)

    logreg = LogisticRegression()  # αρχικοποιηση με default τιμες του logistic regression
    logreg.fit(X_log_reg, y_log_reg)
    y_prediction_pH = logreg.predict(X_test_dataframe)  # predict Y
    print("\n\n\n Y PREDICTION PH\n",y_prediction_pH)

    y_pred_list_pH = y_prediction_pH.values.tolist()
    print("\n list with pH predictions=\n",y_pred_list_pH)

    c1=0
    c2=0
    while l < len(input_list):
        elem_list = input_list[i][8]  # for each element in the inside list aka each row
        if elem_list == 'zero':
            input_list[i].pop(8)
            input_list[i].insert(y_pred_list_pH[c2])
            c2+=1
        l+=1
    new_list_pH=input_list
    print("\nNum of pH changed = ",c2,"\n")

    #ΝΕΑ ΛΙΣΤΑ ΜΕΤΑ ΑΠΟ ΕΠΕΞΕΡΓΑΣΙΑ ΚΑΙ ΣΥΜΠΛΉΡΩΣΗ ΤΩΝ ΚΕΝΏΝ ΜΕ  LOGISTIC REGRESSION
    new_dataframe_w_pH = pd.DataFrame(new_list_pH,
                                 columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                          'pH', 'sulphates', 'alcohol'])
    print("\nX_train_ with avg replace in pH null:\n", new_dataframe)

    return [new_list_pH, new_dataframe_w_pH]




temp33 = remove33pH(X_train)  # call function to empty 33% random fromof ph
X_train_list33 = temp33[0]  # resulting list -33%
X_train33 = temp33[1]  # resulting dataframe -33%

check_work(X_train_list33)

# Ερώτημα b1
tempb1 = b1(X_train33)
X_train_listb1 = tempb1[0]  # list with removed pH element
X_trainb1 = tempb1[1]  # dataframe with removed pH column

# Ερώτημα b2
#tempb2 = b2(X_train_list33)
#X_train_listb2 = tempb2[0]  # list with M.O. at removed pH values
#X_trainb2 = tempb2[1]  # dataframe with M.O. at removed pH values

# To Do Ερώτημα β3 β4
# ΑΛΛΑΓΗ ΣΕ PRECISION 4 ΔΕΚΑΔΙΚΩΝ ΨΗΦΙΩΝ ΣΤΟ FLOAT ΓΙΑ ΠΙΟ ΟΚ ΔΕΔΟΜΕΝΑ

# Ερώτημα b3
# Logistic Regression ειναι binary δηλαδή yes ή no
#print("idfk",X_train_list33)
tempb3 = b3(X_train_list33)
X_train_listb3 = tempb3[0]  # list with logistic regression pH values
X_trainb3 = tempb3[1]  # dataframe with logistic regression removed pH values
