import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from general.baseFileExtractor import get_file_base


sig_feat_herrmannova = ['avgA', 'stdA', 'varA', '25pA', 'skewA', 'minB', 'maxB', 'rangeB', 'sumB', 'avgB', 'stdB',
                        'varB', '25pB', '50pB', 'skewB', 'sumC', 'avgC', 'stdC', 'varC', '25pC', 'kurtC', 'minD',
                        'rangeD', 'sumD', 'avgD', 'stdD', 'varD', '25pD', 'skewD', 'kurtD', 'minE', 'rangeE', 'sumE']

folds = 10
number_of_neighbors = 5
random_state = 100
n_jobs = 1


def read_in_csv_data(source):
    data = pd.read_csv(source, sep=',')
    data = data.sample(frac=1, random_state=random_state)
    labels = data[['class']]
    data.drop(['class'], axis=1, inplace=True)

    return data, labels


def read_in_csv_data_sem_sur_uni(source):
    data = pd.read_csv(source, sep=',')
    sem = pd.DataFrame()
    sur = pd.DataFrame()
    uni = pd.DataFrame()

    data = data.sample(frac=1, random_state=random_state)

    for _ in data.columns[:-1]:
        sem = data[:][data['class'] == 0]
        sur = data[:][data['class'] == 1]
        uni = data[:][data['class'] == 2]

    labels = data[['class']]
    sem.drop(['class'], axis=1, inplace=True)
    sur.drop(['class'], axis=1, inplace=True)
    uni.drop(['class'], axis=1, inplace=True)

    return data, labels, sem, sur, uni


def single_feature_classify_data(X, y, classifier, searched_feat):
    y = y.astype('int')

    # LR
    clf_lr = LogisticRegression(solver='lbfgs', random_state=random_state, max_iter=1000, penalty='l2',
                                fit_intercept=True, multi_class='ovr', n_jobs=n_jobs)

    # RF
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=random_state, criterion='entropy',
                                    max_features='auto',
                                    n_jobs=n_jobs)

    # NB
    clf_nb = GaussianNB()

    # SVM
    clf_svm = SVC(gamma='scale', random_state=random_state)

    # GB
    clf_gb = GradientBoostingClassifier(random_state=random_state, loss='deviance', learning_rate=0.025,
                                        n_estimators=200)

    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=number_of_neighbors, n_jobs=n_jobs)

    # SGD
    clf_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=random_state, tol=None,
                            n_jobs=n_jobs)

    # Model configuration dictionary
    model_config = {'LR': clf_lr, 'RF': clf_rf, 'NB': clf_nb, 'SVM': clf_svm, 'GB': clf_gb, 'KNN': clf_knn,
                    'SGD': clf_sgd}

    def general_predict(model_id, X, y, folds, feature):

        predicted = cross_val_predict(model_config[model_id], X[[feature]], y.values.ravel(), cv=folds)
        cm = confusion_matrix(y, predicted)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return {'C': model_id, 'Feature': feature, 'A': metrics.accuracy_score(y, predicted),
                'F1': metrics.f1_score(y, predicted, average='weighted'),
                '0': cm[0][0], '1': cm[1][1], '2': cm[2][2]}

    def general_cv(model_id, X, y, folds, feature):
        scores = cross_val_score(model_config[model_id], X[[feature]], y.values.ravel(), cv=folds)
        print(scores)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    #print(general_predict(classifier, X, y, folds, searched_feat))

    # LR
    clf_lr = LogisticRegression(solver='lbfgs', random_state=random_state, max_iter=1000, penalty='l2',
                                fit_intercept=True, multi_class='ovr', n_jobs=n_jobs)

    # RF
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=random_state, criterion='entropy',
                                    max_features='auto',
                                    n_jobs=n_jobs)

    # NB
    clf_nb = GaussianNB()

    # SVM
    clf_svm = SVC(gamma='scale', random_state=random_state)

    # GB
    clf_gb = GradientBoostingClassifier(random_state=random_state, loss='deviance', learning_rate=0.025,
                                        n_estimators=200)

    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=number_of_neighbors, n_jobs=n_jobs)

    # SGD
    clf_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=random_state, tol=None,
                            n_jobs=n_jobs)

    # Model configuration dictionary
    model_config = {'LR': clf_lr, 'RF': clf_rf, 'NB': clf_nb, 'SVM': clf_svm, 'GB': clf_gb, 'KNN': clf_knn,
                    'SGD': clf_sgd}

    general_cv(classifier, X, y, folds, searched_feat)


def all_single_feature_classify_data(X, y, classifier):
    y = y.astype('int')
    # LR
    clf_lr = LogisticRegression(solver='lbfgs', random_state=random_state, max_iter=1000, penalty='l2',
                                fit_intercept=True, multi_class='ovr', n_jobs=n_jobs)

    # RF
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=random_state, criterion='entropy',
                                    max_features='auto',
                                    n_jobs=n_jobs)

    # NB
    clf_nb = GaussianNB()

    # SVM
    clf_svm = SVC(gamma='scale', random_state=random_state)

    # GB
    clf_gb = GradientBoostingClassifier(random_state=random_state, loss='deviance', learning_rate=0.025,
                                        n_estimators=200)

    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=number_of_neighbors, n_jobs=n_jobs)

    # SGD
    clf_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=random_state, tol=None,
                            n_jobs=n_jobs)

    # Model configuration dictionary
    model_config = {'LR': clf_lr, 'RF': clf_rf, 'NB': clf_nb, 'SVM': clf_svm, 'GB': clf_gb, 'KNN': clf_knn,
                    'SGD': clf_sgd}

    # the following part of code needs the classifier id
    def all_general_predict(model_id, X, y, folds):
        predicted = cross_val_predict(model_config[model_id], X, y.values.ravel(), cv=folds, n_jobs=n_jobs)
        cm = confusion_matrix(y, predicted)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return {'C': model_id, 'A': metrics.accuracy_score(y, predicted),
               'F1': metrics.f1_score(y, predicted, average='weighted'), '0': cm[0][0], '1': cm[1][1], '2': cm[2][2]}

    # print(all_general_predict(classifier, X, y, folds))

    def general_cv(model_id, X, y, folds):
        scores = cross_val_score(model_config[model_id], X, y.values.ravel(), cv=folds)
        print(scores)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


    # LR
    clf_lr = LogisticRegression(solver='lbfgs', random_state=random_state, max_iter=1000, penalty='l2',
                                fit_intercept=True, multi_class='ovr', n_jobs=n_jobs)

    # RF
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=random_state, criterion='entropy',
                                    max_features='auto',
                                    n_jobs=n_jobs)

    # NB
    clf_nb = GaussianNB()

    # SVM
    clf_svm = SVC(gamma='scale', random_state=random_state)

    # GB
    clf_gb = GradientBoostingClassifier(random_state=random_state, loss='deviance', learning_rate=0.025,
                                        n_estimators=200)

    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=number_of_neighbors, n_jobs=n_jobs)

    # SGD
    clf_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=random_state, tol=None,
                            n_jobs=n_jobs)

    # Model configuration dictionary
    model_config = {'LR': clf_lr, 'RF': clf_rf, 'NB': clf_nb, 'SVM': clf_svm, 'GB': clf_gb, 'KNN': clf_knn,
                    'SGD': clf_sgd}

    general_cv(classifier, X, y, folds)


def restrict_features_to_publication_time(data):
    drop = []
    for feature in data.columns:
        if feature[-1] in ['A', 'C', 'E']:
            drop.append(feature)

    data.drop(drop, axis=1, inplace=True)
    return data


def sem_one_class(vec, dist, classifier, stem, single, searched_feat, restrict_to_publication_time):
    # read in normal distance features for all publications
    # data, labels = read_in_csv_data(get_file_base() + 'extracted_features/OVR/lda_unstemmed_OVR.csv')

    data, labels = read_in_csv_data(get_file_base() + 'extracted_features/' + vec + '_' + dist + '_' +
                                    ('un' if not stem else '') + 'stemmed.csv')

    if single:
        print('SINGLE FEATURE')
        single_feature_classify_data(data, labels, classifier, searched_feat)

    else:
        if restrict_to_publication_time:
            data = restrict_features_to_publication_time(data)

        print('ALL FEATURES')

        all_single_feature_classify_data(data, labels, classifier)
