#!/usr/bin/python

import numpy as np
import math
"""
Function to Select and print k best features from the original feature list
Use scikit-learn's SelectKBest feature selection:
"""


def select_k_best(labels, features, features_list, k):

    ## select best k features on test data generated by stratified shuffle split
    ## returns a list of k best features

    from sklearn.feature_selection import SelectKBest
    from sklearn.model_selection import StratifiedShuffleSplit

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

