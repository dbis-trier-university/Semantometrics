import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.stats import kurtosis, skew
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import wasserstein_distance
import ast
from sklearn.externals import joblib
import pickle
from general.baseFileExtractor import get_survey_u, get_seminal_u, get_uninfluential_u, get_file_base
from computeFeatures.DistancesInYearsAsFeature import make_year


# ----------------------------------------------------------------------------------------------------------------------
# feature computation
def all_features(a, b, c, d, e):
    sets = [a, b, c, d, e]
    features = pd.DataFrame(index=range(12), columns=range(5))

    ds = 0
    while ds < 5:
        # min
        features[ds][0] = np.min(sets[ds][0]) if len(sets[ds][0]) > 1 else 0
        # max
        features[ds][1] = np.max(sets[ds][0]) if len(sets[ds][0]) > 1 else 0
        # range
        features[ds][2] = features[ds][1] - features[ds][0]
        # sum
        features[ds][3] = np.sum(sets[ds][0]) if len(sets[ds][0]) > 1 else 0
        # mean
        features[ds][4] = features[ds][3] / len(sets[ds]) if len(sets[ds]) > 1 else 0
        # variance
        features[ds][6] = np.var(sets[ds][0]) if len(sets[ds][0]) > 1 else 0
        # standard deviation
        features[ds][5] = np.std(sets[ds][0]) if len(sets[ds][0]) > 1 else 0
        # 25 percentile
        features[ds][7] = np.percentile(sets[ds], 25) if len(sets[ds]) > 1 else 0
        # 50 percentile
        features[ds][8] = np.percentile(sets[ds], 50) if len(sets[ds]) > 1 else 0
        # 75 percentile
        features[ds][9] = np.percentile(sets[ds], 75) if len(sets[ds]) > 1 else 0
        # skewness
        features[ds][10] = skew(sets[ds][0]) if len(sets[ds][0]) > 1 else 0
        # kurtosis
        features[ds][11] = kurtosis(sets[ds][0]) if len(sets[ds][0]) > 1 else 0

        ds += 1

    return_string = ''
    for d in range(0, 5):
        for f in range(0, 12):
            return_string += ',' + str(features[d][f])

    return return_string


def write_to_file(file_name, c_v):
    # write to file
    write_file = open(file_name, 'w+')

    write_file.write(
        'class,minA,maxA,rangeA,sumA,avgA,stdA,varA,25pA,50pA,75pA,skewA,kurtA,minB,maxB,rangeB,sumB'
        ',avgB,stdB,varB,25pB,50pB,75pB,skewB,kurtB,minC,maxC,rangeC,sumC,avgC,stdC,varC,25pC,'
        '50pC,75pC,skewC,kurtC,minD,maxD,rangeD,sumD,avgD,stdD,varD,25pD,50pD,75pD,skewD,kurtD'
        ',minE,maxE,rangeE,sumE,avgE,stdE,varE,25pE,50pE,75pE,skewE,kurtE\n')

    for entry in c_v.keys():
        write_file.write(str(0 if entry[1:2] == 'e' else (1 if entry[1:2] == 'u' else 2)) + ' ' + c_v[entry] + '\n')

    write_file.close()


def component_wise_multiplication(x, y):
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)

    return 1 - np.dot(x, y)


def do_component_wise_multiplication(data_set, ds_ct, p):
    print('start ' + str(ds_ct) + ' ' + str(p))
    # calculations of component wise multiplications in sets
    # a
    ct = 0
    a = pd.DataFrame(index=range(len(data_set[ds_ct][1][p]) * len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        rct = 0
        while rct < len(data_set[ds_ct][2][p]):
            a[0][ct] = component_wise_multiplication(data_set[ds_ct][1][p][lct], data_set[ds_ct][2][p][rct])
            ct += 1
            rct += 1
        lct += 1

    # b
    ct = 0
    b = pd.DataFrame(index=range(len(data_set[ds_ct][1][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        b[0][ct] = component_wise_multiplication(data_set[ds_ct][1][p][lct], data_set[ds_ct][0][p])
        ct += 1
        lct += 1

    # c
    ct = 0
    c = pd.DataFrame(index=range(len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        c[0][ct] = component_wise_multiplication(data_set[ds_ct][2][p][lct], data_set[ds_ct][0][p])
        ct += 1
        lct += 1

    # d
    ct = 0
    n = len(data_set[ds_ct][1][p])
    d = pd.DataFrame(index=range(int((n * (n - 1)) / 2)), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        rct = lct + 1
        while rct < len(data_set[ds_ct][1][p]):
            d[0][ct] = component_wise_multiplication(data_set[ds_ct][1][p][lct], data_set[ds_ct][1][p][rct])
            ct += 1
            rct += 1
        lct += 1

    # e
    ct = 0
    n = len(data_set[ds_ct][2][p])
    e = pd.DataFrame(index=range(int((n * (n - 1)) / 2)), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        rct = lct + 1
        while rct < len(data_set[ds_ct][2][p]):
            e[0][ct] = component_wise_multiplication(data_set[ds_ct][2][p][lct], data_set[ds_ct][2][p][rct])
            ct += 1
            rct += 1
        lct += 1

    print('END ' + str(ds_ct) + ' ' + str(p))

    # 1-12 features
    return all_features(a, b, c, d, e)


def do_cosine(data_set, ds_ct, p):
    print('start ' + str(ds_ct) + ' ' + str(p))
    # calculations of distance in sets
    # a
    ct = 0
    a = pd.DataFrame(index=range(len(data_set[ds_ct][1][p]) * len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        rct = 0
        while rct < len(data_set[ds_ct][2][p]):
            a[0][ct] = 1.0 - cosine_similarity(data_set[ds_ct][1][p][lct], data_set[ds_ct][2][p][rct])[0][0]
            if a[0][ct] < 0:
                a[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    # b
    ct = 0
    b = pd.DataFrame(index=range(len(data_set[ds_ct][1][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        b[0][ct] = 1.0 - cosine_similarity(data_set[ds_ct][1][p][lct], data_set[ds_ct][0][p])[0][0]
        if b[0][ct] < 0:
            b[0][ct] = 0
        ct += 1
        lct += 1

    # c
    ct = 0
    c = pd.DataFrame(index=range(len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        c[0][ct] = 1.0 - cosine_similarity(data_set[ds_ct][2][p][lct], data_set[ds_ct][0][p])[0][0]
        if c[0][ct] < 0:
            c[0][ct] = 0
        ct += 1
        lct += 1

    # d
    ct = 0
    n = len(data_set[ds_ct][1][p])
    d = pd.DataFrame(index=range(int((n * (n - 1)) / 2)), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        rct = lct + 1
        while rct < len(data_set[ds_ct][1][p]):
            d[0][ct] = 1.0 - cosine_similarity(data_set[ds_ct][1][p][lct], data_set[ds_ct][1][p][rct])[0][0]
            if d[0][ct] < 0:
                d[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    # e
    ct = 0
    n = len(data_set[ds_ct][2][p])
    e = pd.DataFrame(index=range(int((n * (n - 1)) / 2)), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        rct = lct + 1
        while rct < len(data_set[ds_ct][2][p]):
            e[0][ct] = 1.0 - cosine_similarity(data_set[ds_ct][2][p][lct], data_set[ds_ct][2][p][rct])[0][0]
            if e[0][ct] < 0:
                e[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    print('END ' + str(ds_ct) + ' ' + str(p))

    # 1-12 features
    return all_features(a, b, c, d, e)


def jaccard_sim(x, y):
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)

    return np.dot(x, y) / (np.sum(x) + np.sum(y) - np.dot(x, y))


def do_jaccard(data_set, ds_ct, p):
    print('start ' + str(ds_ct) + ' ' + str(p))
    # calculations of distance in sets
    # a
    ct = 0
    a = pd.DataFrame(index=range(len(data_set[ds_ct][1][p]) * len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        rct = 0
        while rct < len(data_set[ds_ct][2][p]):
            a[0][ct] = 1.0 - jaccard_sim(data_set[ds_ct][1][p][lct], data_set[ds_ct][2][p][rct])
            if a[0][ct] < 0:
                a[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    # b
    ct = 0
    b = pd.DataFrame(index=range(len(data_set[ds_ct][1][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        b[0][ct] = 1.0 - jaccard_sim(data_set[ds_ct][1][p][lct], data_set[ds_ct][0][p])
        if b[0][ct] < 0:
            b[0][ct] = 0
        ct += 1
        lct += 1

    # c
    ct = 0
    c = pd.DataFrame(index=range(len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        c[0][ct] = 1.0 - jaccard_sim(data_set[ds_ct][2][p][lct], data_set[ds_ct][0][p])
        if c[0][ct] < 0:
            c[0][ct] = 0
        ct += 1
        lct += 1

    # d
    ct = 0
    n = len(data_set[ds_ct][1][p])
    d = pd.DataFrame(index=range(int((n * (n - 1)) / 2)), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        rct = lct + 1
        while rct < len(data_set[ds_ct][1][p]):
            d[0][ct] = 1.0 - jaccard_sim(data_set[ds_ct][1][p][lct], data_set[ds_ct][1][p][rct])
            if d[0][ct] < 0:
                d[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    # e
    ct = 0
    n = len(data_set[ds_ct][2][p])
    e = pd.DataFrame(index=range(int((n * (n - 1)) / 2)), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        rct = lct + 1
        while rct < len(data_set[ds_ct][2][p]):
            e[0][ct] = 1.0 - jaccard_sim(data_set[ds_ct][2][p][lct], data_set[ds_ct][2][p][rct])
            if e[0][ct] < 0:
                e[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    print('END ' + str(ds_ct) + ' ' + str(p))

    # 1-12 features
    return all_features(a, b, c, d, e)


# earth mover's distance
def do_wasserstein(data_set, ds_ct, p):
    print('start ' + str(ds_ct) + ' ' + str(p))
    # calculations of distance in sets
    # a
    ct = 0
    a = pd.DataFrame(index=range(len(data_set[ds_ct][1][p]) * len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0

    while lct < len(data_set[ds_ct][1][p]):
        rct = 0
        while rct < len(data_set[ds_ct][2][p]):
            a[0][ct] = wasserstein_distance(data_set[ds_ct][1][p][lct][0], data_set[ds_ct][2][p][rct][0])
            if a[0][ct] < 0:
                a[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    # b
    ct = 0
    b = pd.DataFrame(index=range(len(data_set[ds_ct][1][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        b[0][ct] = wasserstein_distance(data_set[ds_ct][1][p][lct][0], data_set[ds_ct][0][p][0])
        if b[0][ct] < 0:
            b[0][ct] = 0
        ct += 1
        lct += 1

    # c
    ct = 0
    c = pd.DataFrame(index=range(len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        c[0][ct] = wasserstein_distance(data_set[ds_ct][2][p][lct][0], data_set[ds_ct][0][p][0])
        if c[0][ct] < 0:
            c[0][ct] = 0
        ct += 1
        lct += 1

    # d
    ct = 0
    n = len(data_set[ds_ct][1][p])
    d = pd.DataFrame(index=range(int((n * (n - 1)) / 2)), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        rct = lct + 1
        while rct < len(data_set[ds_ct][1][p]):
            d[0][ct] = wasserstein_distance(data_set[ds_ct][1][p][lct][0], data_set[ds_ct][1][p][rct][0])
            if d[0][ct] < 0:
                d[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    # e
    ct = 0
    n = len(data_set[ds_ct][2][p])
    e = pd.DataFrame(index=range(int((n * (n - 1)) / 2)), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        rct = lct + 1
        while rct < len(data_set[ds_ct][2][p]):
            e[0][ct] = wasserstein_distance(data_set[ds_ct][2][p][lct][0], data_set[ds_ct][2][p][rct][0])
            if e[0][ct] < 0:
                e[0][ct] = 0
            ct += 1
            rct += 1
        lct += 1

    print('END ' + str(ds_ct) + ' ' + str(p))

    # 1-12 features
    return all_features(a, b, c, d, e)


def task_bert(survey_hlp, seminal_hlp, uninfluential_hlp):
    seminal_p = {}
    seminal_x = {}
    seminal_y = {}

    ct = 0
    for p in seminal_hlp['seminal']:
        seminal_p[ct] = np.array(p['bert']).reshape(1, -1)

        seminal_x[ct] = {}
        seminal_y[ct] = {}

        ct_x = 0
        for o in p['ref']:
            seminal_x[ct][ct_x] = np.array(o).reshape(1, -1)
            ct_x += 1

        ct_y = 0
        for i in p['cit']:
            seminal_y[ct][ct_y] = np.array(i).reshape(1, -1)
            ct_y += 1

        ct += 1

    survey_p = {}
    survey_x = {}
    survey_y = {}

    ct = 0
    for p in survey_hlp['survey']:
        survey_p[ct] = np.array(p['bert']).reshape(1, -1)

        survey_x[ct] = {}
        survey_y[ct] = {}

        ct_x = 0
        for o in p['ref']:
            survey_x[ct][ct_x] = np.array(o).reshape(1, -1)
            ct_x += 1

        ct_y = 0
        for i in p['cit']:
            survey_y[ct][ct_y] = np.array(i).reshape(1, -1)
            ct_y += 1

        ct += 1

    uninfluential_p = {}
    uninfluential_x = {}
    uninfluential_y = {}

    ct = 0
    for p in uninfluential_hlp['uninfluential']:
        uninfluential_p[ct] = np.array(p['bert']).reshape(1, -1)

        uninfluential_x[ct] = {}
        uninfluential_y[ct] = {}

        ct_x = 0
        for o in p['ref']:
            uninfluential_x[ct][ct_x] = np.array(o).reshape(1, -1)
            ct_x += 1

        ct_y = 0
        for i in p['cit']:
            uninfluential_y[ct][ct_y] = np.array(i).reshape(1, -1)
            ct_y += 1

        ct += 1

    return [[seminal_p, seminal_x, seminal_y, 'sem '], [survey_p, survey_x, survey_y, 'surv '],
            [uninfluential_p, uninfluential_x, uninfluential_y, 'uni ']]


def order_publications(unordered_p, unordered_x, unordered_y, data):
    ordered_p = {}
    ordered_x = {}
    ordered_y = {}

    ct = 0

    # iterate through keys in data to get order of publications
    for x in data:
        ordered_p[ct] = unordered_p[x['key']]
        ordered_x[ct] = unordered_x[x['key']]
        ordered_y[ct] = unordered_y[x['key']]
        ct += 1

    return ordered_p, ordered_x, ordered_y


def read_in_json_lda_data(pub_type, data):
    pub_p = {}
    pub_x = {}
    pub_y = {}

    for p in data[pub_type]:
        pub_p[p['key']] = [ast.literal_eval(p['lda'])]
        pub_x[p['key']] = []
        pub_y[p['key']] = []

        for pref in p['ref']:
            pub_x[p['key']].append([ast.literal_eval(pref)])

        for pcit in p['cit']:
            pub_y[p['key']].append([ast.literal_eval(pcit)])

    return pub_p, pub_x, pub_y


def task_lda(use_stemming):
    with open(get_survey_u(), encoding='latin-1') as s:
        survey_hlp = json.load(s)
    with open(get_seminal_u(), encoding='latin-1') as s:
        seminal_hlp = json.load(s)
    with open(get_uninfluential_u(), encoding='latin-1') as s:
        uninfluential_hlp = json.load(s)

    with open(get_file_base() + 'lda_data/sem_lda_' + ('un' if not use_stemming else '') + 'stemmed.json', 'r') as \
            sem_file:
        sem = json.load(sem_file)
    with open(get_file_base() + 'lda_data/sur_lda_' + ('un' if not use_stemming else '') + 'stemmed.json', 'r') as \
            sur_file:
        sur = json.load(sur_file)
    with open(get_file_base() + 'lda_data/uni_lda_' + ('un' if not use_stemming else '') + 'stemmed.json', 'r') as \
            sur_file:
        uni = json.load(sur_file)

    # seminal
    unordered_seminal_p, unordered_seminal_x, unordered_seminal_y = read_in_json_lda_data('seminal', sem)
    # survey
    unordered_survey_p, unordered_survey_x, unordered_survey_y = read_in_json_lda_data('survey', sur)
    # uninfluential
    unordered_uninfluential_p, unordered_uninfluential_x, unordered_uninfluential_y = \
        read_in_json_lda_data('uninfluential', uni)

    seminal_hlp = seminal_hlp['seminal']
    survey_hlp = survey_hlp['survey']
    uninfluential_hlp = uninfluential_hlp['uninfluential']

    # matching of ordering of publication with sur/sem/uni_stemmed/unstemmed-data
    seminal_p, seminal_x, seminal_y = order_publications(unordered_seminal_p, unordered_seminal_x, unordered_seminal_y,
                                                         seminal_hlp)
    survey_p, survey_x, survey_y = order_publications(unordered_survey_p, unordered_survey_x, unordered_survey_y,
                                                      survey_hlp)
    uninfluential_p, uninfluential_x, uninfluential_y = order_publications(unordered_uninfluential_p,
                                                                           unordered_uninfluential_x,
                                                                           unordered_uninfluential_y,
                                                                           uninfluential_hlp)

    return [[seminal_p, seminal_x, seminal_y, 'sem '], [survey_p, survey_x, survey_y, 'surv '],
            [uninfluential_p, uninfluential_x, uninfluential_y, 'uni ']]


def make_features(metric, task, this_use_stemming):
    data_set = None

    if metric not in ['cos', 'jac', 'emd', 'ipd', 'dist']:
        print('Metric ' + metric + ' unknown.')
        return
    if task not in ['tfidf', 'd2v', 'bert', 'lda', 'year']:
        print('Task ' + task + ' unknown.')
        return

    print('Using ' + metric + ' on ' + ('un' if not this_use_stemming else '') + 'stemmed ' + task + ' vectors.')

    if metric == 'dist' and task == 'year':
        make_year()
        return

    if task == 'tfidf':
        with open(get_file_base() + 'tfidf_data/tfidf_' + ('un' if not this_use_stemming else '') + 'stemmed.sav',
                  'rb') as f:
            data_set = joblib.load(f)

        # for p in range(0, len(data_set[2][0])):
        #    data_set[2][0][p] = [data_set[2][0][p]]

        # for p in range(0, len(data_set[2][1])):
        #    for x in range(0, len(data_set[2][1][p])):
        #       data_set[2][1][p][x] = [data_set[2][1][p][x]]

        # for p in range(0, len(data_set[2][2])):
        #    for x in range(0, len(data_set[2][2][p])):
        #       data_set[2][2][p][x] = [data_set[2][2][p][x]]

    if task == 'd2v':
        with open(get_file_base() + 'd2v_data/d2v_unstemmed.pickle', 'rb') as f:
            data_set = pickle.load(f)

    if task == 'bert':
        with open(get_file_base() + 'bert_data/sur_bert_unstemmed.json', encoding='latin-1') as s:
            survey_hlp = json.load(s)
        with open(get_file_base() + 'bert_data/sem_bert_unstemmed.json', encoding='latin-1') as s:
            seminal_hlp = json.load(s)
        with open(get_file_base() + 'bert_data/uni_bert_unstemmed.json', encoding='latin-1') as s:
            uninfluential_hlp = json.load(s)

        data_set = task_bert(survey_hlp, seminal_hlp, uninfluential_hlp)

    if task == 'lda':
        data_set = task_lda(this_use_stemming)

    ds_ct = 0
    executor = ThreadPoolExecutor(max_workers=64)
    completed_vecs = {}

    # data_set[][0] -> P
    # data_set[][1] -> X, references
    # data_set[][2] -> Y, citations
    while ds_ct < len(data_set):
        p = 0
        while p < len(data_set[ds_ct][0]):
            futures = None
            if metric == 'emd':
                futures = executor.submit(do_wasserstein, data_set, ds_ct, p)
            if metric == 'cos':
                futures = executor.submit(do_cosine, data_set, ds_ct, p)
            if metric == 'jac':
                futures = executor.submit(do_jaccard, data_set, ds_ct, p)
            if metric == 'ipd':
                futures = executor.submit(do_component_wise_multiplication, data_set, ds_ct, p)

            completed_vecs[data_set[ds_ct][3] + str(p)] = futures.result()
            p += 1
        ds_ct += 1

    write_to_file(get_file_base() + 'extracted_features/' + task + '_' + metric + '_' +
                  ('un' if not this_use_stemming else '') + 'stemmed.csv', completed_vecs)
