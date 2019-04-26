from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from  sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score


#This is code for gender Classificaton using Various classifiers and Suggestes the best one to use.

#  X is a collection of height, weight, and shoe size combinations.
#  Y contains the gender labels associated with each combination.
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38],
     [154, 54, 37],[166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43], [168, 75, 41], [168, 77, 41]]


Y= ['male','male','female','female','male','male','female','female','female','male','male','female','female']

dec_clf = tree.DecisionTreeClassifier()
dec_clf = dec_clf.fit(X,Y)

test_data = [[190, 70, 43],[154, 75, 38],[181,65,40]] #Data for testing
test_labels = ['male','female','male'] # Expected results

dec_prediction = dec_clf.predict(test_data)

print("Using Decision tree  ",dec_prediction)

# USING RANDOM FOREST CLASSIFIER

# create and configure model

rnf_clf = RandomForestClassifier(n_estimators=100)
rnf_clf = rnf_clf.fit(X,Y)

rnf_prediction = rnf_clf.predict(test_data)

print("Using Random forest  ",rnf_prediction)

#USING LogisticRegression

lor_clf = LogisticRegression(solver='lbfgs')
lor_clf = lor_clf.fit(X,Y)

lor_prediction = lor_clf.predict(test_data)

print("Using logisticRegression  ",lor_prediction)

# Using SVM

svm_clf = SVC(gamma='scale').fit(X,Y)
svm_prediction = svm_clf.predict(test_data)

print("Using Support vector  ",svm_prediction)

# test label is compared with outcome of algorithms for measurement of acccuracy

dec_accuracy = accuracy_score(dec_prediction,test_labels)
rnf_accuracy = accuracy_score(rnf_prediction,test_labels)
lor_accuracy = accuracy_score(lor_prediction,test_labels)
svm_accuracy = accuracy_score(svm_prediction,test_labels)


classifiers= ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Support vecto']

accuracy = np.array([dec_accuracy,rnf_accuracy,lor_accuracy,svm_accuracy])
max_accuracy_index = np.argmax(accuracy) #Returns maximum accuracy's index

print(classifiers[max_accuracy_index])

print(classifiers[max_accuracy_index] +"is the best gender Classifier for this set of data ")