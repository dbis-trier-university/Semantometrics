import pandas as pd
import json
from evaluateMethod.ClassificationOVR import all_single_feature_classify_data
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from general.baseFileExtractor import get_file_base, get_seminal_u, get_survey_u, get_uninfluential_u

number_of_neighbors = 5
random_state = 100
n_jobs = -1

# LR
clf_lr = LogisticRegression(solver='lbfgs', random_state=random_state, max_iter=1000, penalty='l2',
                            fit_intercept=True, multi_class='ovr', n_jobs=n_jobs)

# RF
clf_rf = RandomForestClassifier(n_estimators=100, random_state=random_state, criterion='entropy', max_features='auto',
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
clf_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=random_state, tol=None, n_jobs=n_jobs)

# Model configuration dictionary
model_config = {'LR': clf_lr, 'RF': clf_rf, 'NB': clf_nb, 'SVM': clf_svm, 'GB': clf_gb, 'KNN': clf_knn,
                'SGD': clf_sgd}


def read_in_csv_data(source):
    return pd.read_csv(source, sep=',')


def find_equal():
    with open(get_seminal_u(), encoding='latin-1') as s:
        seminal_hlp = json.load(s)
        seminal_hlp = seminal_hlp['seminal']

    with open(get_survey_u(), encoding='latin-1') as s:
        survey_hlp = json.load(s)
        survey_hlp = survey_hlp['survey']

    with open(get_uninfluential_u(), encoding='latin-1') as s:
        uninfluential_hlp = json.load(s)
        uninfluential_hlp = uninfluential_hlp['uninfluential']

    lengths_sem = {}

    for entry_id in range(0, len(seminal_hlp)):
        abs_length = len(seminal_hlp[entry_id]['abs'].split())
        if abs_length not in lengths_sem:
            lengths_sem[abs_length] = [entry_id]
        else:
            lengths_sem[abs_length].append(entry_id)

    lengths_sur = {}

    for entry_id in range(0, len(survey_hlp)):
        abs_length = len(survey_hlp[entry_id]['abs'].split())
        if abs_length not in lengths_sur:
            lengths_sur[abs_length] = [entry_id]
        else:
            lengths_sur[abs_length].append(entry_id)

    lengths_uni = {}

    for entry_id in range(0, len(uninfluential_hlp)):
        abs_length = len(uninfluential_hlp[entry_id]['abs'].split())
        if abs_length not in lengths_uni:
            lengths_uni[abs_length] = [entry_id]
        else:
            lengths_uni[abs_length].append(entry_id)

    found = []

    for entry in lengths_sem:
        if entry in lengths_sur and entry in lengths_uni:
            found.append(entry)

    sem_ids = []
    sur_ids = []
    uni_ids = []
    for entry in found:
        num = min(len(lengths_sem[entry]), len(lengths_sur[entry]), len(lengths_uni[entry]))

        for i in range(0, num):
            sem_ids.append(lengths_sem[entry][i])
            sur_ids.append(lengths_sur[entry][i])
            uni_ids.append(lengths_uni[entry][i])

    return sem_ids, sur_ids, uni_ids


def prepare_data(data, s_sem, s_sur, s_uni):
    new_data = pd.DataFrame(data=data)

    print('anz sem ' + str(len(s_sem)))
    print('anz sur ' + str(len(s_sur)))
    print('anz uni ' + str(len(s_uni)))

    for short in range(0, 660):
        if short not in s_sem:
            new_data.drop(short, inplace=True)

    for short in range(0, 660):
        if short not in s_sur:
            new_data.drop(660 + short, inplace=True)

    for short in range(0, 660):
        if short not in s_uni:
            new_data.drop(2*660 + short, inplace=True)

    new_data = new_data.sample(frac=1, random_state=random_state)

    labels = new_data[['class']]
    new_data.drop(['class'], axis=1, inplace=True)

    return new_data, labels


def main():
    # read in normal distance features for all publications
    data = read_in_csv_data(get_file_base() + 'extracted_features/d2v_cos_unstemmed.csv')

    s_sem, s_sur, s_uni = find_equal()

    data, labels = prepare_data(data, s_sem, s_sur, s_uni)

    for model_id in model_config:
        all_single_feature_classify_data(data, labels, model_id)


if __name__ == '__main__':
    main()
