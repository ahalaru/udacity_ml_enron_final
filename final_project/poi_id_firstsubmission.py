#!/usr/bin/python

import sys
import pickle
import pandas

from feature_format import featureFormat, targetFeatureSplit
from pickler import dump_classifier_and_data

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

## All the features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive','restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']  #omitted email_address as the value is a string
features_list= ['poi'] + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###Convert to panda dataframe for ease of investigation of the dataset
df = pandas.DataFrame.from_records(list(data_dict.values()))

#people= pandas.Series(list(data_dict.keys()))
#feature_names = list(df.columns.values)

print "Number of Records" ,df.shape
print "Number of POI in DataSet: ", (df['poi'] == 1).sum()
print "Number of non-POI in Dataset: ", (df['poi'] == 0).sum()


###Remove outliers

#LOCKHART EUGENE has all values blank and the other two are not a person and hence irrelevant in model to find POI

outliers= ['TOTAL' , 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for name in outliers:
    del data_dict[name]

df = pandas.DataFrame.from_records(list(data_dict.values()))
print "Number of records after outlier removal" ,df.shape

### Create new feature(s)

#New feature 'percentage_sent_to_poi' gives proportion of total from_messages sent to POI

import math

for key,value in data_dict.items():
    from_msg = float(value['from_messages'])
    from_this_to_poi = float(value['from_this_person_to_poi'])
    if (not math.isnan(from_msg) and from_msg !=0 )and not math.isnan(from_this_to_poi):
        ratio = from_this_to_poi/from_msg
    value['percentage_sent_to_poi'] = ratio

features_list.append('percentage_sent_to_poi')

#New feature 'percentage_received_from_poi gives proportion of total to_messages received from POI

for key,value in data_dict.items():
    to_msg = float(value['to_messages'])
    from_poi_to_this = float(value['from_poi_to_this_person'])
    if (not math.isnan(to_msg) and to_msg !=0 )and not math.isnan(from_poi_to_this):
        ratio = from_poi_to_this/to_msg
    value['percentage_received_from_poi'] = ratio

features_list.append('percentage_received_from_poi')


### Store to my_dataset for easy export below.
my_dataset = data_dict

## Function to Select and print k best features from the original feature list
##Use scikit-learn's SelectKBest feature selection:

def get_k_best(enron_data, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """

    from sklearn.feature_selection import SelectKBest
    data = featureFormat(enron_data, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())

    return k_best_features

# get K-best features
target_label = 'poi'
num_features = 9 # 9 best features
best_features = get_k_best(data_dict, features_list, num_features)
my_feature_list = [target_label] + best_features.keys()


## Task 4: Trying a variety of classifiers
### I name my classifier clf for easy export below.

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)

'''Other classifiers tried

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=30, )

#GridSearchCV
from sklearn.model_selection import GridSearchCV
tree_para = {'max_depth':[4,5,6,7,8,9,10], 'min_samples_split' : [2,5,10,50]}
clf = GridSearchCV(DecisionTreeClassifier(random_state=0), tree_para)

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf= GaussianNB()

'''

### Extract features and labels from dataset for performance testing

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

data = featureFormat(data_dict, my_feature_list)
labels, features = targetFeatureSplit(data)

###Using StratifiedShuffleSplit using 1000 iterations to shuffle and split the data

from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 1000, random_state=42)

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv:
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    for ii in train_idx:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_idx:
        features_test.append(features[jj])
        labels_test.append(labels[jj])

    ### fit the classifier using training set, and test on test set

    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break
try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
    precision = 1.0 * true_positives / (true_positives + false_positives)
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    print clf
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision=5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives,
                                       true_negatives)
    print ""
except:
    print "Got a divide by zero when trying out:", clf
    print "Precision or recall may be undefined due to a lack of true positive predicitons."

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_feature_list)
