

# Introduction to Machine Learning - Final Project
## Building Person-Of-Interest Identifier in Enron fraud case. 
#### By Arundathi Acharya

## Project Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

In this project I put my Machine Learning skills to use to build **person-of-interest(POI)** identifier based on financial and email data made public as a result of the Enron scandal. The goal of this project is to use this public data along with a given hand-generated list of **person-of-interest(individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity)** to build and train the identifier(Machine learning model) to identify a POI.


## Setting up Python environment and modules


```python
import sys
import pickle
import pandas
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

```

## Dataset and feature introduction

Enron email and financial data is combined into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

**financial features**: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

**email features**: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

**POI label**: [‘poi’] (boolean, represented as integer)



```python
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
###Convert to panda dataframe for ease of investigation of the dataset
df = pandas.DataFrame.from_records(list(data_dict.values()))
print "Number of Records" ,df.shape
print "Number of POI in DataSet: ", (df['poi'] == 1).sum()
print "Number of non-POI in Dataset: ", (df['poi'] == 0).sum()
```

Number of Records (146, 21)
Number of POI in DataSet:  18
Number of non-POI in Dataset:  128

The dictionary potentially has information about 146 people involved with Enron stored as dictionary keys. All the 21 feature names along with relevant values for each person is stored in python dictionary data_dict. The key feature of interest is if a person is Person-of-interest(POI is 1) or no(POI is 0). There are only 18 POI and the rest 128 are non POI. Hence it is a very unbalanced dataset. So it is worth noting that there will be very few training points for POI to train the classifier.

## Missing Values and Nan

The pdf containing the financial information about the 146 people involved with Enron has a lot of missing values for many of the financial features. When the financial information is stored in the data_dict, the missing values are stored as Nan. The financial information for a feature missing for any person is likely to be because it is not relevant for the person and can hence be safely marked as zero. The function featureformat() does this programatically and marks any Nan with 0.0 while converting the data dictionary to numpy array of features, the format required by most of the sklearn modules. 

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)


## Outlier investigation

The approach to handle outliers in financial data of Enron differs from other scenarios where we usually remove the outliers. This is a case of corporate fraud and our goal is to build a model which can identify a POI who are involved in it. Invariably most of the POIs were holding high ranking high paying posts in Enron including the CEO , chairman and such. So this is a special case where we are specifically interested in outliers and removing them will undermine the quality of data by losing the vital information needed to achieve the goal of this project, specially knowing that there are only 18 POIs. Since there are only 146 records, this case calls for manual inspection of the data.

Nevertheless, I inspect the finanical data manually to check for any potential skewing and I see 3 candidates for removal. The last record called 'THE TRAVEL AGENCY IN THE PARK' stands out as it is not refering to any person. Also the row for 'TOTAL' of all values is a definite candidate for removal as rest of the records are for individual people. LOCKHART EUGENE has all values blank and hence will be removed from the data dictionary. 


```python
### Task 2: Remove outliers
outliers= ['TOTAL' , 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for name in outliers:
    del data_dict[name]
    
print "Number of records after outlier removal" ,df.shape
```

Number of records after outlier removal (143, 21)

## Feature Selection


```python
# All original 

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive','restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']  #omitted string email address
features_list= ['poi'] + financial_features + email_features

```

I want to be able to choose from all the features listed above in the POI identifier I plan to build, except the feature 'email_address' in email feature,  which is the text_string. The feature 'POI' is the target feature that we want to predict. The function targetFeatureSplit() is used to split data into targets and features as separate lists.


```python
def targetFeatureSplit( data ):
#given a numpy array like the one returned from featureFormat, separate out the first feature and put it into its own list 

```

#### Building new features

I build 2 new features based on the intuition that there might be stronger email connections between POIs than between POIs and non POIs.

1> New feature **'percentage_sent_to_poi'** gives what proportion of total from_messages sent from this person is being sent to a POI.

2> Similarly new feature **'percentage_received_from_poi'** gives what proportion of total to_messages received by this person is being received from a POI


```python
for key,value in data_dict.items():
    from_msg = float(value['from_messages'])
    from_this_to_poi = float(value['from_this_person_to_poi'])
    if (not math.isnan(from_msg) and from_msg !=0 )and not math.isnan(from_this_to_poi):
        ratio = from_this_to_poi/from_msg
    value['percentage_sent_to_poi'] = ratio

features_list.append('percentage_sent_to_poi')

for key,value in data_dict.items():
    to_msg = float(value['to_messages'])
    from_poi_to_this = float(value['from_poi_to_this_person'])
    if (not math.isnan(to_msg) and to_msg !=0 )and not math.isnan(from_poi_to_this):
        ratio = from_poi_to_this/to_msg
    value['percentage_received_from_poi'] = ratio

features_list.append('percentage_received_from_poi')
```

#### Setting up automatic feature selection function _SelectKBest_

I use the univariate feature selection function SelectKBest to automatically pick K best features 


```python
##Use scikit-learn's SelectKBest feature selection to select K best features from a given feature list

def get_k_best(enron_data, features_list, k):
    #runs scikit-learn's SelectKBest feature selection
    #returns dict where keys=features, values=scores
    
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
```

Selecting 10 best features and getting the lables and corresponding features ready to be used in a classifier


```python
# get K-best features
target_label = 'poi'
num_features = 10  # 10 best features
best_features = get_k_best(data_dict, features_list, num_features)
my_feature_list = [target_label] + best_features.keys()

#get data for labels and features for corresponding K-best features

data = featureFormat(data_dict, my_feature_list)
labels, features = targetFeatureSplit(data)
```

#### Feature scaling

I plan to use Decision Trees and Naive Bayes classifier to begin with and hence will not use feature scaling for now. Generally, feature scaling is needed when the estimator is operating in a coordinate plane. The reason for this is that we do not want one feature to be weighted disproportionately to all of the others simply because it has bigger values.If we do not scale your features, then a difference of 100 in a feature with a range of 0 to a million will seem very large next to features that have a range of 0 to 1. The distance in the coordinate plane will far outweigh the smaller features. With scaling, this problem is eliminated.

Gaussian Naive Bayes and Decision Trees(and their ensembles) do not operate in the coordinate plain, so they do not need feature scaling. Decision Tree splits are based on each feature and independent from one another, and will only split at points that make sense given the data.Naive Bayes generates probabilities for all of the features individually, assuming no correlation, and so does not need feature scaling.


## Algorithms

To build a POI identifier, supervised learning classification algorithm is befitting, as the model must classify if a person is POI or no. The data under consideration has clearly labeled features and targets and the output is discrete. I choose to start with simple classifier like Gaussian Naive Bayes and then work my way up with Decision Trees and ensembles. With small amount of data in a large number of dimensions and relative sparseness of information, SVM doesnot seem to be a good fit for the classifier.

### Setting up tester to evaluate the classifier performance

First I set up a tester to compute scores for the variety of classifier 'clf' that I will build and try. The performance metrics that I am particularly interested is Accuracy , Precision and Recall. Since the size of the data is small and unbalanced, I use the StratifiedShuffleSplit function of sklearn.cross_validation to shuffle the data over each iteration while splitting into test and train sets. Stratified Shuffle Split is important since it makes sure the ratio of POI and non-POI is the same during training and testing.


```python
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

```

### _Setting up Naive Bayes Classifer_

I plan to use a variety of choice of features viz all original features, all features along with newly built feature, features picked by SeleckKBest and compare performances of the Naive Bayes algorithm with it. First I set up the Naive Bayes classifier and then record the performances with a variety of feature combinations.


```python
from sklearn.naive_bayes import GaussianNB
clf= GaussianNB()
```

#### Wtih all original features

['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
 'shared_receipt_with_poi']

GaussianNB(priors=None)
	Accuracy: 0.72760	Precision: 0.20898	Recall: 0.37450	F1: 0.26827	F2: 0.32329
	Total predictions: 15000	True positives:  749	False positives: 2835	False negatives: 1251	True negatives: 10165



#### With all original features plus two new features created 'percentage_sent_to_poi' and 'percentage_received_from_poi'

['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
 'shared_receipt_with_poi', 'percentage_sent_to_poi', 'percentage_received_from_poi']


GaussianNB(priors=None)

	Accuracy: 0.72760	Precision: 0.20898	Recall: 0.37450	F1: 0.26827	F2: 0.32329
	Total predictions: 15000	True positives:  749	False positives: 2835	False negatives: 1251	True negatives: 10165

This took me by surprise as there was no change in performance with the additional new features. So next I use the SelectKBest and try varying the number of features chosen 

#### With 10 best features from SelectKBest

10 best features: ['salary', 'total_payments', 'loan_advances', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

GaussianNB(priors=None)

	Accuracy: 0.81520	Precision: 0.29854	Recall: 0.28600	F1: 0.29213	F2: 0.28842
	Total predictions: 15000	True positives:  572	False positives: 1344	False negatives: 1428	True negatives: 11656

With automatic feature selection the performance of the classifier improved significantly. Now I test the performance with fewer number of features

#### With 9 best features from SelectKBest

9 best features: ['salary', 'total_payments', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

GaussianNB(priors=None)

	Accuracy: 0.83560	Precision: 0.35930	Recall: 0.29750	F1: 0.32549	F2: 0.30810
	Total predictions: 15000	True positives:  595	False positives: 1061	False negatives: 1405	True negatives: 11939


With 1 less feature the performance further imporoved. The recall is very close to the desired threshold of 0.3. I am hopeful reducing the number of features will give the desired performance. 

#### With 8 best features from SelectKBest

8 best features: ['salary', 'total_payments', 'bonus', 'total_stock_value', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

GaussianNB(priors=None)

	Accuracy: 0.84100	Precision: 0.37870	Recall: 0.30050	F1: 0.33510	F2: 0.31345
	Total predictions: 15000	True positives:  601	False positives:  986	False negatives:1399	True negatives:12014

Gaussian Naive Bayes seems to have satisfactory precision (above the required threshold of 0.3) and recall with the 8 features. 

### _Setting up Decision Tree Classifier_

I next setup and try the Decision Tree Classifier with the various feature combinations like I tried above. 


```python
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)
```

#### With all original features

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
            
	Accuracy: 0.79573	Precision: 0.21941	Recall: 0.20800	F1: 0.21355	F2: 0.21019
	Total predictions: 15000	True positives:  416	False positives: 1480	False negatives: 1584	True negatives: 11520


#### With all original features plus two new features created 'percentage_sent_to_poi' and 'percentage_received_from_poi'

['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'percentage_sent_to_poi', 'percentage_received_from_poi']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
            
	Accuracy: 0.79740	Precision: 0.23125	Recall: 0.22350	F1: 0.22731	F2: 0.22501
	Total predictions: 15000	True positives:  447	False positives: 1486	False negatives: 1553	True negatives: 11514

Unlike the Naive Bayes, the decision tree classifier showed slight increase in its performance with the additional new features. But it is not very significant so I try the classifier with features selected by SelectKBest

#### With 10 best features from SelectKBest

10 best features: ['salary', 'total_payments', 'loan_advances', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
            
	Accuracy: 0.81193	Precision: 0.29465	Recall: 0.29450	F1: 0.29457	F2: 0.29453
	Total predictions: 15000	True positives:  589	False positives: 1410	False negatives: 1411	True negatives: 11590

The performance of the classifier is significantly imroved here with 10 best features

#### With 9 best features from SelectKBest

9 best features: ['salary', 'total_payments', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
            
	Accuracy: 0.81373	Precision: 0.30728	Recall: 0.31650	F1: 0.31182	F2: 0.31461
	Total predictions: 15000	True positives:  633	False positives: 1427	False negatives: 1367	True negatives: 11573

The performance further improved with 1 less feature and both percision and recall are greater than the threshold of 0.3

#### With 8 best features from SelectKBest

8 best features: ['salary', 'total_payments', 'bonus', 'total_stock_value', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
            
	Accuracy: 0.80380	Precision: 0.27145	Recall: 0.28000	F1: 0.27566	F2: 0.27825
	Total predictions: 15000	True positives:  560	False positives: 1503	False negatives: 1440	True negatives: 11497


Further reducing the number of features does not improve the performance, infact the performance metrics have decreased. Hence 9 features selected by SelectKBest give the best performance for the DecisionTree Classfier.

### Parameter Tuning

Once an algorithm is chosen for a task, we use Parameter tuning to try various options for the parameters of the algorithm to see what combination of value for each parameter gives the best performance for the algorithm at hand. Parameter tuning is important to make the most of the algorithm chosen , or in other words, select the best parameters to optimize the performance of the algorithm. This can be achieved by trying various options manually and see how the performance varies for each of the trials. A better and faster approach will be to use tuning functions like  GridSearchCV where we can specify a list of options to be tried for every parameter. For the parameters where a value is not specified, the default value is used. 

#### Using GridSearchCV to tune the Decision Tree with 9 best features from SelectKBest

Now that I am convinced that 9 best features from SelectKBest give the best performance for the DecisionTree Classifier with the default values, I use the GridSearchCV function to tune the algorithm by trying a range of different values fro the parameters 'max_depth' and 'min_samples_split'.  

Also, since the number of data is relatively small, I expect that Stratified Shuffle Split combined with Grid Search CV can be used with acceptable training time. For a larger dataset, Randomized Search CV might be more suitable.


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
tree_para = {'max_depth':[4,5,6,7,8,9,10], 'min_samples_split' : [2,5,10,50]}
clf = GridSearchCV(DecisionTreeClassifier(random_state = 0), tree_para)
```

Using GridSearchCV is a very time expensive process. Hence I reduced the number of folds in my tester function StratifiedShuffleSplit from 1000 to 100. 

#### Using 100 folds in StratifiedShuffleSplit and using GridSearchCV for DecisionTree

9 best features: ['salary', 'total_payments', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

GridSearchCV(cv=None, error_score='raise',
       estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best'),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'min_samples_split': [2, 5, 10, 50], 'max_depth': [4, 5, 6, 7, 8, 9, 10]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
       scoring=None, verbose=0)
       
	Accuracy: 0.83400	Precision: 0.28319	Recall: 0.16000	F1: 0.20447	F2: 0.17525
	Total predictions: 1500	True positives:   32	False positives:   81	False negatives:  168	True negatives: 1219

#### Using 100 folds in StratifiedShuffleSplit without using GridSearchCV and just default DecisonTree

9 best features: ['salary', 'total_payments', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
            
	Accuracy: 0.81333	Precision: 0.31308	Recall: 0.33500	F1: 0.32367	F2: 0.33037
	Total predictions: 1500	True positives:   67	False positives:  147	False negatives:  133	True negatives: 1153

I drop using GridsearchCV as it does not boost the performance and the time tradeoff does not make it worth it.

#### Using the ensemble AdaBoostClassifier to boost the performance of DecisionTree with default settings


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=30, )
```

##### With 9 best features from SelectKBest

9 best features: ['salary', 'total_payments', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best'),
          learning_rate=1.0, n_estimators=50, random_state=30)
          
	Accuracy: 0.81033	Precision: 0.28715	Recall: 0.28500	F1: 0.28607	F2: 0.28543
	Total predictions: 15000	True positives:  570	False positives: 1415	False negatives: 1430	True negatives: 11585

##### With 10 best features from SelectKBest

10 best features: ['salary', 'total_payments', 'loan_advances', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best'),
          learning_rate=1.0, n_estimators=50, random_state=30)

	Accuracy: 0.81207	Precision: 0.29657	Recall: 0.29850	F1: 0.29753	F2: 0.29811
	Total predictions: 15000	True positives:  597	False positives: 1416	False negatives: 1403	True negatives: 11584



##### With 11 best features from SelectKBest

11 best features: ['salary', 'total_payments', 'loan_advances', 'bonus', 'percentage_sent_to_poi', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']

AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best'),
          learning_rate=1.0, n_estimators=50, random_state=30)

	Accuracy: 0.81773	Precision: 0.31295	Recall: 0.30700	F1: 0.30994	F2: 0.30817
	Total predictions: 15000	True positives:  614	False positives: 1348	False negatives: 1386	True negatives: 11652


Using AdaBoost Classifier with DecisionTreeClassifier as base estimator we reach the evalution metric threshold of 0.3 with 11 features. 

### Thoughts on Validation and performance

Validation is performed to ensure that a machine learning algorithm generalizes well. A classic mistake is over-fitting, where the model is trained and performs very well on the training dataset, but markedly worse on the cross-validation and test datasets. I believe the algorithm chosen and the classifier built was trained over 1000 folds of shuffling of test and train data using **cross-validation function StratifiedShuffleSplit**. So I believe the performance metrics above are well validated.


The main evaluation metrics used to judge the performance of the classifier is **'Precision' and 'Recall'**. Since the data at hand is unbalanced, (small proprotion of POIs compared to the non-POIs), accuracy is not a significant evaluation metric.

Precision of the classifier measures the percentage of records classified as POIs by the classifier that are actually POIs. For example, a precision of 0.32 suggests if a person is classified as a POI by the classifier, there is a 32% probability that the person is actually a POI. It is the ratio of true positive to all positives marked by a classifier.

Recall of the classifier measures the percentage of true POIs that is correctly classified as POI by the classifier.For example, a precision of 0.3 suggests that of all the POIs (18 in our case), 30% of them are correctly classified as POI by the classifier. It is the ratio of true positives to actual number of true values. 

### Best pick so far

The following two classifiers gave precision and recall both above 0.3. Even the accuracy measured up pretty close

**DecisionTreeClassifier(Default parameters, 9 Best features)**    Accuracy: 0.81373	 Precision: 0.30728	  Recall: 0.31650

**AdaboostClassifier(With base DecisionTree , 11 Best features)**  Accuracy: 0.81773	 Precision: 0.31295	  Recall: 0.30700


I chose the DecisionTreeClassifier with Default Parameters along with 9 Best features selected by SelectKBest as the algorithm-feature combination in my final project. The reason is because it gave the best values for perfomance metrics with minimum number of features compared to any other algorithm that I tested. 

### Final Analysis

So far , the feature selection process on this project was performed on the entire dataset i.e the get_k_best() function took the entire enron data set to test the features for performance. Now I try a better approach which would combine some sort of cross-validation with the selectKBest process (a stratified shuffle split to be specific). 

To do this, I set up the cross-validation split of the data set, and perform feature selection on the training set of each fold. The selectKBest algorithm gets an F-score for each feature and chooses the k features with the best F-scores. When combined with cross-validation, an F-score will be produced for each feature and fold. The best features can then be considered those with the best average F-scores over all folds in the cross validation. 

I use this idea to write the function select_k_best() and will use it instead of earlier function get_k_best()


```python
def select_k_best(enron_data, features_list, k):

    ## select best k features on test data generated by stratified shuffle split
    ## returns a list of k best features

    from sklearn.feature_selection import SelectKBest
    from sklearn.model_selection import StratifiedShuffleSplit

    data = featureFormat(enron_data, features_list)
    labels, features = targetFeatureSplit(data)

    sss = StratifiedShuffleSplit(n_splits=1000, random_state=42)

    # We will include all the features into variable best_features, then group by their
    # occurrences.
    features_scores = {}

    for i_train, i_test in sss.split(features, labels):
        features_train, features_test = [features[i] for i in i_train], [features[i] for i in i_test]
        labels_train, labels_test = [labels[i] for i in i_train], [labels[i] for i in i_test]

        # fit selector to training set
        selector = SelectKBest(k=k)
        selector.fit(features_train, labels_train)

        # Get scores of each feature:
        sel_features = np.array(features_list[1:])[selector.get_support()]
        sel_list = []
        for i in range(len(sel_features)):
            sel_list.append([sel_features[i], selector.scores_[i], selector.pvalues_[i]])

        # Fill to feature_scores dictionary
        for feat, score, pvalue in sel_list:
            if feat not in features_scores:
                features_scores[feat] = {'scores': [], 'pvalues': []}
            if not math.isnan(score):
                features_scores[feat]['scores'].append(score)
            if not math.isnan(pvalue):
                features_scores[feat]['pvalues'].append(pvalue)

    # Average scores and pvalues of each feature
    features_scores_l = []  # tuple of feature name, avg scores, avg pvalues
    for feat in features_scores:
        features_scores_l.append((
            feat,
            np.mean(features_scores[feat]['scores']),
            np.mean(features_scores[feat]['pvalues'])
        ))

    # Sort by scores and display
    import operator
    sorted_feature_scores = sorted(features_scores_l, key=operator.itemgetter(1), reverse=True)
    sorted_feature_scores_str = ["{}, {}, {}".format(x[0], x[1], x[2]) for x in sorted_feature_scores]

    print "feature, score, pvalue"
    print sorted_feature_scores_str

    # Return k best features based on the sorted list
    return [sorted_feature_scores[i][0] for i in range(k)]
```

Now I test this new function and check how performance of the DecisionTree classifier varies for different number of features

#### with 9 best features from select_k_best ()

9 best features: ['from_this_person_to_poi', 'salary', 'restricted_stock', 'other', 'long_term_incentive', 'from_poi_to_this_person', 'to_messages', 'total_stock_value', 'deferred_income']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')
            
	Accuracy: 0.78160	Precision: 0.18164	Recall: 0.18200	F1: 0.18182	F2: 0.18193
	Total predictions: 15000	True positives:  364	False positives: 1640	False negatives: 1636	True negatives: 11360


With this new approach, the performance of DecisionTreeClassifier with 9 best features is significantly poor. I try with increased number of features next

#### With 10 best features

10 best features: ['percentage_sent_to_poi', 'other', 'long_term_incentive', 'shared_receipt_with_poi', 'salary', 'deferred_income', 'from_this_person_to_poi', 'restricted_stock', 'from_poi_to_this_person', 'exercised_stock_options']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')

	Accuracy: 0.81400	Precision: 0.30250	Recall: 0.30250	F1: 0.30250	F2: 0.30250
	Total predictions: 15000	True positives:  605	False positives: 1395	False negatives: 1395	True negatives: 11605


With 10 features, the DecisionTree Classifier performed better and I am surprised to see the new feature I created viz 'percentage_sent_to_poi' chosen as one of the 10 best feature. I now try with 11 features

#### With 11 best features

11 best features: ['other', 'to_messages', 'deferred_income', 'salary', 'from_poi_to_this_person', 'long_term_incentive', 'restricted_stock', 'exercised_stock_options', 'shared_receipt_with_poi', 'expenses', 'loan_advances']

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')

	Accuracy: 0.79053	Precision: 0.24093	Recall: 0.26550	F1: 0.25262	F2: 0.26019
	Total predictions: 15000	True positives:  531	False positives: 1673	False negatives: 1469	True negatives: 11327

With 11 best features the classifier performance did not improve and instead the evaluation metrics are lower than that with 10 features. So now I try with lower number of features and try with 8 best features

#### With 8 best features

** 8 best features: ['percentage_sent_to_poi', 'shared_receipt_with_poi', 'restricted_stock', 'salary', 'total_stock_value', 'expenses', 'other', 'from_poi_to_this_person']**

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best')

	Accuracy: 0.81660	Precision: 0.31584	Recall: 0.32200	F1: 0.31889	F2: 0.32075
	Total predictions: 15000	True positives:  644	False positives: 1395	False negatives: 1356	True negatives: 11605 

** With 8 best features the DecisionTreeClassifier gave the best performance compared to all the trials so far. The required threshold of 0.3 is achieved for both 'Precision' and 'Recall'. It is nice to get better performance with lesser number of features and hence I choose this as the algorithm and feature combination for the final analysis. **
