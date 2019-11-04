#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import pprint
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# print([len(v) for v in features])
# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)

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
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

n_components = 7
for i in range(1,len(features_list)):

    pca = PCA(svd_solver='randomized',n_components=i, whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = {
             'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    pred = clf.predict(X_test_pca)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    print "n_components: ", i
    print "precision: ", precision
    print "recall: ", recall
    print "F1score: ", F1, "\n\n"
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()



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

dump_classifier_and_data(clf, my_dataset, features_list)
