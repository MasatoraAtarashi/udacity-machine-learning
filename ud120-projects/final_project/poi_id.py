#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','poi','salary', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
# features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
# features_list = ['poi', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
# features_list = ['poi', 'shared_receipt_with_poi', 'from_this_person_to_poi','bonus', 'total_stock_value']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
# data_dict.pop('LAY KENNETH L')
# data_dict.pop('SKILLING JEFFREY K')
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# addtional_data = featureFormat(my_dataset, ['to_messages','from_messages'], sort_keys = True)
# print len(my_dataset)
print len(features)
# print len(addtional_data)
# a = input()
# print([len(v) for v in features])
# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)

# for idx,point in enumerate(features):
#     bonus = point[0]
#     total_stock_value = point[1]
#     if labels[idx] == 1:
#         matplotlib.pyplot.scatter( bonus, total_stock_value, c='g' )
#     else:
#         matplotlib.pyplot.scatter( bonus, total_stock_value, c='r' )
#
# matplotlib.pyplot.xlabel(features_list[1])
# matplotlib.pyplot.ylabel(features_list[2])
# matplotlib.pyplot.show()

# ratio_to_from_messages = []
# for point in features:
#     if np.isnan(point[9]) or np.isnan(point[11]):
#         ratio_to_from_messages.append(0)
#     else:
#         ratio_to_from_messages.append(point[9] / float(point[11]))
#
# ratio_to_from_messages = [1000 if np.isinf(i) else i for i in ratio_to_from_messages]
# ratio_to_from_messages = [0 if np.isnan(i) else i for i in ratio_to_from_messages]
# pprint.pprint(ratio_to_from_messages)
# print([len(v) for v in features])
# print(sum(len(v) for v in features))
# new_features = []
# for i in range(len(features)):
#     a = []
#     for j in range(len(features[i])):
#         # 9,11
#         if j == 9 or j == 11:
#             continue
#         a.append(features[i][j])
#     a.append(ratio_to_from_messages[i])
#     new_features.append(a)
#
# features = new_features
# for i in data_dict.keys():
#     print data_dict[i]['total_stock_value']

# for point in features:
#     bonus = point[3]
#     total_stock_value = point[5]
#     matplotlib.pyplot.scatter( bonus, total_stock_value )
#
# matplotlib.pyplot.xlabel(features_list[3])
# matplotlib.pyplot.ylabel(features_list[5])
# matplotlib.pyplot.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

n_components = 4
# for i in range(1,len(features_list)):

pca = PCA(svd_solver='randomized',n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train_pca = scaler.transform(X_train)
# X_test_pca = scaler.transform(X_test)

# print "SVM\n"
param_grid = {
         'C': [0.1,0.01,1,3,5,10,15,1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,1.0],
          }
# score = 'f1'
# clf = GridSearchCV(
#     SVC(kernel='rbf', class_weight='balanced'),
#     param_grid,
#     cv=5,
#     scoring=metrics.make_scorer(metrics.f1_score, pos_label = 1, average="binary")
#     )
best_C = 0
best_gamma = 0
best_f1 = 0
for c in param_grid["C"]:
    for g in param_grid["gamma"]:
        clf = SVC(kernel='rbf', class_weight='balanced',C=c,gamma=g)
        clf = clf.fit(X_train_pca, y_train)
        pred = clf.predict(X_test_pca)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        F1 = 2 * (precision * recall) / (precision + recall)
        print "c: ", c ,"gamma: ", g, "f1: ",F1
        if F1 > best_f1:
            best_f1 = F1
            best_C = c
            best_gamma = g
print "\n"
print "c: ",best_C,"bestgamma: ",best_gamma, "bestf1: ",best_f1
clf = SVC(kernel='rbf', class_weight='balanced',C=best_C,gamma=best_gamma)
clf = clf.fit(X_train_pca, y_train)
# pprint.pprint(clf.grid_scores_)
# print clf.best_params_
pred = clf.predict(X_test_pca)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
precision = tp / float(tp + fp)
recall = tp / float(tp + fn)
F1 = 2 * (precision * recall) / (precision + recall)
# print "n_components: ", i
print "poi num: ", len(pred[pred == 1])
print "precision: ", precision
print "recall: ", recall
print "F1score: ", F1
print "test accuracy", accuracy_score(y_test, pred), "\n"

# pred = clf.predict(X_train_pca)
# tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()
# precision = tp / float(tp + fp)
# recall = tp / float(tp + fn)
# F1 = 2 * (precision * recall) / (precision + recall)
# # print "n_components: ", i
# print "poi num: ", len(pred[pred == 1])
# print "precision: ", precision
# print "recall: ", recall
# print "F1score: ", F1
# print "train accuracy", accuracy_score(y_train, pred),"\n"

# from sklearn.model_selection import learning_curve
# train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='rbf'), X_train, y_train, train_sizes=[15,30,45,66], cv=None)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(valid_scores, axis=1)
# test_scores_std = np.std(valid_scores, axis=1)
# plt.grid()
#
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1,
#                  color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
# p1 = plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#          label="Training score")
# p2 = plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#          label="Cross-validation score")
# # p1 = plt.plot(train_sizes, train_scores, color="red")
# # p2 = plt.plot(train_sizes, valid_scores, color="blue")
# plt.legend((p1[0], p2[0]), ("train_scores", "valid_scores"), loc=2)
# plt.show()
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# print "decisiontree\n"
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(min_samples_split=40)
# clf = clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
# tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
# precision = tp / float(tp + fp)
# recall = tp / float(tp + fn)
# F1 = 2 * (precision * recall) / (precision + recall)
# print "precision: ", precision
# print "recall: ", recall
# print "F1score: ", F1, "\n"

# from sklearn.svm import SVC
# for c in [1, 10.0, 100, 1000, 10000]:
#     clf = SVC(C=c, kernel='rbf')
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     from sklearn.metrics import confusion_matrix
#     tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
#     precision = tp / float(tp + fp)
#     recall = tp / float(tp + fn)
#     F1 = 2 * (precision * recall) / (precision + recall)
#     print "parameter: ", c
#     print "precision: ", precision
#     print "recall: ", recall
#     print "F1score: ", F1, "\n\n"

# 2 is best maybe
# for sp in [2, 4, 8, 16, 32, 40, 100]:
#     from sklearn import tree
#     clf = tree.DecisionTreeClassifier(min_samples_split=sp)
#     clf = clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     from sklearn.metrics import confusion_matrix
#     tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
#     precision = tp / float(tp + fp)
#     recall = tp / float(tp + fn)
#     F1 = 2 * (precision * recall) / (precision + recall)
#     print "parameter: ", c
#     print "precision: ", precision
#     print "recall: ", recall
#     print "F1score: ", F1, "\n\n"



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# return F1,recall,precision
dump_classifier_and_data(clf, my_dataset, features_list)
