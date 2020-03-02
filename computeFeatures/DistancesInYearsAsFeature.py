import json
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from concurrent.futures import ThreadPoolExecutor
from general.baseFileExtractor import get_survey_s, get_seminal_s, get_uninfluential_s, get_file_base


# ----------------------------------------------------------------------------------------------------------------------
# feature computation
def all_features(a, b, c, d, e):
    sets = [a, b, c, d, e]
    features = pd.DataFrame(index=range(12), columns=range(5))

    ds = 0
    while ds < 5:
        # min
        features[ds][0] = np.min(sets[ds][0])
        # max
        features[ds][1] = np.max(sets[ds][0])
        # range
        features[ds][2] = features[ds][1] - features[ds][0]
        # sum
        features[ds][3] = np.sum(sets[ds][0])
        # mean
        features[ds][4] = features[ds][3] / len(sets[ds])
        # variance
        features[ds][6] = np.var(sets[ds][0])
        # standard deviation
        features[ds][5] = np.std(sets[ds][0])
        # 25 percentile
        features[ds][7] = np.percentile(sets[ds], 25)
        # 50 percentile
        features[ds][8] = np.percentile(sets[ds], 50)
        # 75 percentile
        features[ds][9] = np.percentile(sets[ds], 75)
        # skewness
        features[ds][10] = skew(sets[ds][0])
        # kurtosis
        features[ds][11] = kurtosis(sets[ds][0])

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


def do_difference(data_set, ds_ct, p):
    print('start ' + str(ds_ct) + ' ' + str(p))
    # calculations of component wise multiplications in sets
    # a
    ct = 0
    a = pd.DataFrame(index=range(len(data_set[ds_ct][1][p]) * len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        rct = 0
        while rct < len(data_set[ds_ct][2][p]):
            a[0][ct] = np.abs(data_set[ds_ct][1][p][lct] - data_set[ds_ct][2][p][rct])
            ct += 1
            rct += 1
        lct += 1

    # b
    ct = 0
    b = pd.DataFrame(index=range(len(data_set[ds_ct][1][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][1][p]):
        b[0][ct] = np.abs(data_set[ds_ct][1][p][lct] - data_set[ds_ct][0][p])
        ct += 1
        lct += 1

    # c
    ct = 0
    c = pd.DataFrame(index=range(len(data_set[ds_ct][2][p])), columns=range(1))
    lct = 0
    while lct < len(data_set[ds_ct][2][p]):
        c[0][ct] = np.abs(data_set[ds_ct][2][p][lct] - data_set[ds_ct][0][p])
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
            d[0][ct] = np.abs(data_set[ds_ct][1][p][lct] - data_set[ds_ct][1][p][rct])
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
            e[0][ct] = np.abs(data_set[ds_ct][2][p][lct] - data_set[ds_ct][2][p][rct])
            ct += 1
            rct += 1
        lct += 1

    print('END ' + str(ds_ct) + ' ' + str(p))

    # 1-12 features
    return all_features(a, b, c, d, e)


def task_year(survey_hlp, seminal_hlp, uninfluential_hlp):
    seminal_p = {}
    seminal_x = {}
    seminal_y = {}

    ct = 0
    for p in seminal_hlp['seminal']:
        seminal_p[ct] = p['year']

        seminal_x[ct] = {}
        seminal_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            seminal_x[ct][ct_x] = ref['year']
            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            seminal_y[ct][ct_y] = cit['year']
            ct_y += 1

        ct += 1

    survey_p = {}
    survey_x = {}
    survey_y = {}

    ct = 0
    for p in survey_hlp['survey']:
        survey_p[ct] = p['year']

        survey_x[ct] = {}
        survey_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            survey_x[ct][ct_x] = ref['year']
            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            survey_y[ct][ct_y] = cit['year']
            ct_y += 1

        ct += 1

    uninfluential_p = {}
    uninfluential_x = {}
    uninfluential_y = {}

    ct = 0
    for p in uninfluential_hlp['uninfluential']:
        uninfluential_p[ct] = p['year']

        uninfluential_x[ct] = {}
        uninfluential_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            uninfluential_x[ct][ct_x] = ref['year']
            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            uninfluential_y[ct][ct_y] = cit['year']
            ct_y += 1

        ct += 1

    return [[seminal_p, seminal_x, seminal_y, 'sem '], [survey_p, survey_x, survey_y, 'surv '],
            [uninfluential_p, uninfluential_x, uninfluential_y, 'uni ']]


def make_year():
    # read in
    with open(get_survey_s(), encoding='latin-1') as s:
        survey_hlp = json.load(s)
    with open(get_seminal_s(), encoding='latin-1') as s:
        seminal_hlp = json.load(s)
    with open(get_uninfluential_s(), encoding='latin-1') as s:
        uninfluential_hlp = json.load(s)

    data_set = task_year(survey_hlp, seminal_hlp, uninfluential_hlp)

    ds_ct = 0
    executor = ThreadPoolExecutor(max_workers=64)
    completed_vecs = {}

    while ds_ct < len(data_set):
        p = 0
        while p < len(data_set[ds_ct][0]):
            futures = executor.submit(do_difference, data_set, ds_ct, p)

            completed_vecs[data_set[ds_ct][3] + str(p)] = futures.result()
            p += 1
        ds_ct += 1

    write_to_file(get_file_base() + 'extracted_features/years_dist_unstemmed.csv', completed_vecs)
    write_to_file(get_file_base() + 'extracted_features/years_dist_stemmed.csv', completed_vecs)
