#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
# print len(y_test)
# print len(pred[pred == 1])
print "accuracy", accuracy_score(y_test, pred)
# print "all zero accuracy", accuracy_score(y_test, ([0]*len(y_test)))
# true_positive = 0;
# false_positive = 0;
# false_negative = 0;
# for idx,item in enumerate(y_test):
#     if item == 1 and pred[idx] == 1:
#         true_positive += 1
#     if item == 0 and pred[idx] == 1:
#         false_positive += 1
#     if item == 1 and pred[idx] == 0:
#         false_negative += 1
# # print "true positive count", count
# print "recall: ", true_positive / float(true_positive + false_negative)
# print "precision: ", true_positive / float(true_positive + false_positive)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
print "tp: ", tp
print "fp: ", fp
print "tn: ", tn
print "fn: ", fn
# print tp, true_positive
# print fp, false_positive
# print fn, false_negative
print "precision: ", tp / float(tp + fp)
print "recall: ", tp / float(tp + fn)
from sklearn.metrics import classification_report
print(classification_report(true_labels, predictions))
