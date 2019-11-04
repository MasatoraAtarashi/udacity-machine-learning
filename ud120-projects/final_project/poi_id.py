#!/usr/bin/python

import sys
import pickle
import math
import pprint
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.model_selection import train_test_split
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_poi_ratio','from_poi_ratio','shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

keys = my_dataset.keys()
for key in keys:
    to_poi_ratio = \
        float(my_dataset[key]['from_this_person_to_poi']) / \
        float(my_dataset[key]['from_messages'])
    from_poi_ratio = \
        float(my_dataset[key]['from_poi_to_this_person']) / \
        float(my_dataset[key]['to_messages'])

    my_dataset[key]['to_poi_ratio'] = "NaN" if math.isnan(to_poi_ratio) else to_poi_ratio
    my_dataset[key]['from_poi_ratio'] = "NaN" if math.isnan(from_poi_ratio) else from_poi_ratio
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# feature scaling
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(features)
# features = scaler.transform(features)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(features)
features = scaler.transform(features)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
# clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
# clf = LogisticRegression(C=1,random_state=0, solver='lbfgs')
# clf = clf.fit(X_train,y_train)
# pred = clf.predict(X_test)

n_components = 6
# for i in range(2,20):
#     print "n_components: ", n_components
#     print "split: ", i
#     f1s = []
#     for j in range(1,100):
#         X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)
#         pca = PCA(svd_solver='randomized',n_components=n_components, whiten=True).fit(X_train)
#         X_train_pca = pca.transform(X_train)
#         X_test_pca = pca.transform(X_test)
#         clf = tree.DecisionTreeClassifier(min_samples_split=i)
#         clf = clf.fit(X_train, y_train)
#         pred = clf.predict(X_test)
#
#         f1s.append(f1_score(y_test, pred, average='binary'))
#     print sum(f1s) / float(len(f1s))
#     f1s = []
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)
pca = PCA(svd_solver='randomized',n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
clf = tree.DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print f1_score(y_test, pred, average='binary')
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
