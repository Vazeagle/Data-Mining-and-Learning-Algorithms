import pandas as pd
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time
import tensorflow
from keras.callbacks import EarlyStopping

start_time = time.time()    # Αρχή χρονομέτρησης.

df = pd.read_csv("onion-or-not.csv")
dftext = df.text
dflabel = df.label

stemmer = PorterStemmer()    # Αρχικοποίηση stemming.
tf_idf = TfidfVectorizer()  # Αρχικοποίηση tf-idf.
estop = EarlyStopping(monitor='val_loss', mode='min', verbose=1)   # Αρχικοποίηση early stopping.
stopWords = set(stopwords.words('english'))    # Θέσιμο γλώσσας για αρκετά κοινές λέξεις χωρίς πληροφορίες(stop words).

text_matrix = dftext.to_numpy()
print(text_matrix, "\n\n")
wordvectors = []
for strings in range(len(text_matrix)):     # Προσθήκη κάθε πρότασης του text_matrix.
    for word in word_tokenize(text_matrix[strings]):    # Προσθήκη κάθε λέξης από κάθε πρόταση.
        text_matrix[strings] = text_matrix[strings].replace(word, stemmer.stem(word))
        if word in stopWords:  # Έλεγχος για stop words.
            text_matrix[strings] = text_matrix[strings].replace(word, "")


print("\n")
print(text_matrix, "\n\n")

x = tf_idf.fit(text_matrix)
print(x.vocabulary_)


x = tf_idf.transform(text_matrix)   # Εκτέλεση μεταμόρφωσης με tf-idf.
dataf = pd.DataFrame(x.toarray(), columns=tf_idf.get_feature_names())
print("\n\n\n")

dataf.insert(len(dataf.columns), "labelz", dflabel, True)   # Εισαγωγή των label στο dataframe dataf.
print(dataf, "\n\n")
print("Size of dataframe: \t\t", dataf.size, "\n\n")

X_train, X_test, y_train, y_test = train_test_split(dataf.loc[:, dataf.columns != 'labelz'], dataf.labelz, test_size=0.25)   # Χωρισμός για εκπαίδευση και δοκιμή.

print(X_train)

model = tensorflow.keras.models.Sequential()    # Αρχικοποίηση μοντέλου.
model.add(tensorflow.keras.layers.Flatten())    # Flatten του μοντέλου σε μορφή 1D array.
model.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))    # Προσθήκη επιπέδου.
model.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))    # Προσθήκη επιπέδου.
model.add(tensorflow.keras.layers.Dense(2, activation=tensorflow.nn.softmax))   # 2 πιθανές απαντήσεις (0 / 1).
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   # Σύνταξη μοντέλου.
model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=10, batch_size=16, verbose=1, validation_data=(X_test.to_numpy(), y_test.to_numpy()), callbacks=[estop])  # Ταίριασμα μοντέλου.

val_loss, val_acc = model.evaluate(X_test, y_test)  # Αξιολόγηση μοντέλου.
print("\n\nLoss: \t\t", val_loss, "\nAccuracy: \t", val_acc, "\n\n")

predictions = model.predict(X_test.to_numpy())  # Προβλέψεις μοντέλου.

predictions_list = list()
for k in range(len(predictions)):   # Για κάθε πρόβλεψη.
    predictions_list.append(numpy.argmax(predictions[k]))   # Πρόσθεσε την πρόβλεψη στο predictions_list.


print("F1 Score: \t\t\t", round(f1_score(y_test, predictions_list, average='micro'), 4))
print("Precision Score: \t", round(precision_score(y_test, predictions_list, average='micro'), 4))
print("Recall Score: \t\t", round(recall_score(y_test, predictions_list, average='micro'), 4), "\n\n")

print("--- %s seconds ---" % (time.time() - start_time))    # Τέλος χρονομέτρησης και εκτύπωση.