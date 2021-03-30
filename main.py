import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as plt


import sklearn
from setuptools.dist import sequence
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier

data = pd.read_excel(r'R_data.xlsx')

#turns the data frame to an array
date_df = data[["tarih", "temsilci","Key3","Key4","Key5","Key6","Key7","Key8","Key9","Key10","Key11","Key12","Key13"]]
date = np.array(date_df)

data = data.sample(frac=0.1, random_state = 48) # sample size of 10% of the dataset can comment out for use of all dataset

Y = data.iloc[:, 13].values #correct output values(0 or 1) to compare with

#turns dataframe to an array
x_df = data[["Key3","Key4","Key5","Key6","Key7","Key8","Key9","Key10","Key11","Key12","Key13"]]  # input data
X = np.array(x_df)

#test_size, .3 of dataset is to test set and the .7 is to the test set if test_size = .3 ###
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = .3) # split into 4 arrays, Xtrain = X data

#Decision Tree
#classifier=DecisionTreeClassifier(max_depth=4 ) #, class_weight={0:2.5,1:8}  # this is the decision tree that came with the code from the teacher
#classifier.fit(x_train,y_train)
#predicted=classifier.predict(x_test) #predict output


#K neighbors
model = KNeighborsClassifier(n_neighbors=3) #this is the K Neighbors
model.fit(x_train,y_train)
predicted = model.predict(x_test)

#print results
for x in range(len(x_test)):
    print("Predicted:", predicted[x], " Data:", x_test[x], "Actual:", y_test[x])

    for i in range(len(date)):
        if x_test[x][0] in date[i] and x_test[x][1] in date[i] and x_test[x][2] in date[i]:
            print(" date:", date[i][0], " ", date[i][1], "\n")
            break
        continue

#measure accuracy
DT = sklearn.metrics.accuracy_score(y_test, predicted) * 100
print("\nThe accuracy score using the DecisionTreeClassifier : ",DT)
print('precision')
 #Precision = TP / (TP + FP) (Where TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative).
precision = precision_score(y_test, predicted,average='micro', pos_label=1)
print(precision_score(y_test, predicted, average='micro', pos_label=1))

print("\npredicted values :\n",predicted)
print("\nReal values :\n",y_test)

