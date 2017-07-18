#!/usr/bin/python

import pickle
import pandas


from feature_format import featureFormat, targetFeatureSplit
from pickler import dump_classifier_and_data
from select_features import select_k_best
from trainer import train_classfier

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

## All the features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',\
                      'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',\
                      'long_term_incentive','restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']  #omitted email_address as the value is a string
feature_list= ['poi'] + financial_features + email_features

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

#Remove outliers

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

feature_list.append('percentage_sent_to_poi')

#New feature 'percentage_received_from_poi gives proportion of total to_messages received from POI

for key,value in data_dict.items():
    to_msg = float(value['to_messages'])
    from_poi_to_this = float(value['from_poi_to_this_person'])
    if (not math.isnan(to_msg) and to_msg !=0 )and not math.isnan(from_poi_to_this):
        ratio = from_poi_to_this/to_msg
    value['percentage_received_from_poi'] = ratio

feature_list.append('percentage_received_from_poi')


### Store to my_dataset for easy export below.
my_dataset = data_dict
data = featureFormat(data_dict, feature_list)
labels, features_all = targetFeatureSplit(data)

# get K-best features
target_label = 'poi'
num_features = 8
best_k_features = select_k_best(labels, features_all, feature_list, num_features)
print "\n{0} best features: {1}\n".format(num_features, best_k_features)

#create the feature list with poi and the selected bext features
selected_features = [target_label] + best_k_features


## Task 4: Trying a variety of classifiers
### I name my classifier clf for easy export below.

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

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
data = featureFormat(data_dict, selected_features)
labels, features = targetFeatureSplit(data)

#Train the classifer
train_classfier(clf,labels,features)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, selected_features)


