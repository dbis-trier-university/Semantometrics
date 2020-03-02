import re
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from concurrent.futures import ThreadPoolExecutor
from computeFeatures.FeaturesFromEmbedding import all_features, write_to_file
import pickle
import random
from general.baseFileExtractor import get_file_base


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


def task_d2v(survey_hlp, seminal_hlp, uninfluential_hlp):
    # model 0 : dbow; model 1: dm
    doc2vec_model = Doc2Vec.load(get_file_base() + 'models/doc2vec_model_1')

    seminal_p = {}
    seminal_x = {}
    seminal_y = {}

    ct = 0
    for p in seminal_hlp['seminal']:
        doc2vec_model.random.seed(0)
        hlp = doc2vec_model.infer_vector(p['abs'].split(), alpha=0.025, steps=20)
        seminal_p[ct] = hlp.reshape(1, -1)

        seminal_x[ct] = {}
        seminal_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            doc2vec_model.random.seed(0)
            hlp = doc2vec_model.infer_vector(ref['abs'].split(), alpha=0.025, steps=20)
            seminal_x[ct][ct_x] = hlp.reshape(1, -1)
            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            doc2vec_model.random.seed(0)
            hlp = doc2vec_model.infer_vector(cit['abs'].split(), alpha=0.025, steps=20)
            seminal_y[ct][ct_y] = hlp.reshape(1, -1)
            ct_y += 1

        ct += 1

    survey_p = {}
    survey_x = {}
    survey_y = {}

    ct = 0
    for p in survey_hlp['survey']:
        surv_abs = []
        for w in re.compile('[^a-zA-Z0-9]+').split(p['abs']):
            surv_abs.append(w.lower())

        doc2vec_model.random.seed(0)
        hlp = doc2vec_model.infer_vector(surv_abs, alpha=0.025, steps=20)
        survey_p[ct] = hlp.reshape(1, -1)

        survey_x[ct] = {}
        survey_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            abs_o = []
            for w in re.compile('[^a-zA-Z0-9]+').split(ref['abs']):
                abs_o.append(w.lower())

            doc2vec_model.random.seed(0)
            hlp = doc2vec_model.infer_vector(abs_o, alpha=0.025, steps=20)
            survey_x[ct][ct_x] = hlp.reshape(1, -1)
            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            abs_i = []
            for w in re.compile('[^a-zA-Z0-9]+').split(cit['abs']):
                abs_i.append(w.lower())

            doc2vec_model.random.seed(0)
            hlp = doc2vec_model.infer_vector(abs_i, alpha=0.025, steps=20)
            survey_y[ct][ct_y] = hlp.reshape(1, -1)
            ct_y += 1

        ct += 1

    uninfluential_p = {}
    uninfluential_x = {}
    uninfluential_y = {}

    ct = 0
    for p in uninfluential_hlp['uninfluential']:
        surv_abs = []
        for w in re.compile('[^a-zA-Z0-9]+').split(p['abs']):
            surv_abs.append(w.lower())

        doc2vec_model.random.seed(0)
        hlp = doc2vec_model.infer_vector(surv_abs, alpha=0.025, steps=20)
        uninfluential_p[ct] = hlp.reshape(1, -1)

        uninfluential_x[ct] = {}
        uninfluential_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            abs_o = []
            for w in re.compile('[^a-zA-Z0-9]+').split(ref['abs']):
                abs_o.append(w.lower())

            doc2vec_model.random.seed(0)
            hlp = doc2vec_model.infer_vector(abs_o, alpha=0.025, steps=20)
            uninfluential_x[ct][ct_x] = hlp.reshape(1, -1)
            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            abs_i = []
            for w in re.compile('[^a-zA-Z0-9]+').split(cit['abs']):
                abs_i.append(w.lower())

            doc2vec_model.random.seed(0)
            hlp = doc2vec_model.infer_vector(abs_i, alpha=0.025, steps=20)
            uninfluential_y[ct][ct_y] = hlp.reshape(1, -1)
            ct_y += 1

        ct += 1

    return [[seminal_p, seminal_x, seminal_y, 'sem '], [survey_p, survey_x, survey_y, 'surv '],
            [uninfluential_p, uninfluential_x, uninfluential_y, 'uni ']]


def generate_robust_ds(data_set):
    robust_data_set = data_set

    for p in range(0, 660):
        # seminal p
        # calculate reference candidates to be deleted
        ran_del = set()

        while len(ran_del) < 5:
            ran_del.add(random.randrange(0, len(data_set[0][1][p])))

        # delete 5 references
        new_p = []
        for ref in range(len(data_set[0][1][p])):
            if ref not in ran_del:
                new_p.append(data_set[0][1][p][ref])

        data_set[0][1][p] = new_p

        # calculate citation candidates to be deleted
        ran_del = set()

        while len(ran_del) < 5:
            ran_del.add(random.randrange(0, len(data_set[0][2][p])))

        # delete 5 citations
        new_p = []
        for cit in range(len(data_set[0][2][p])):
            if cit not in ran_del:
                new_p.append(data_set[0][2][p][cit])

        data_set[0][2][p] = new_p

        # survey p
        # calculate reference candidates to be deleted
        ran_del = set()

        while len(ran_del) < 5:
            ran_del.add(random.randrange(0, len(data_set[1][1][p])))

        # delete 5 references
        new_p = []
        for ref in range(len(data_set[1][1][p])):
            if ref not in ran_del:
                new_p.append(data_set[1][1][p][ref])

        data_set[1][1][p] = new_p

        # calculate citation candidates to be deleted
        ran_del = set()

        while len(ran_del) < 5:
            ran_del.add(random.randrange(0, len(data_set[1][2][p])))

        # delete 5 citations
        new_p = []
        for cit in range(len(data_set[1][2][p])):
            if cit not in ran_del:
                new_p.append(data_set[1][2][p][cit])

        data_set[1][2][p] = new_p

        # uninfluential p
        # calculate reference candidates to be deleted
        ran_del = set()

        while len(ran_del) < 2:
            ran_del.add(random.randrange(0, len(data_set[2][1][p])))

        # delete 2 references
        new_p = []
        for ref in range(len(data_set[2][1][p])):
            if ref not in ran_del:
                new_p.append(data_set[2][1][p][ref])

        data_set[2][1][p] = new_p

        # calculate citation candidates to be deleted
        ran_del = set()

        while len(ran_del) < 2:
            ran_del.add(random.randrange(0, len(data_set[2][2][p])))

        # delete 2 citations
        new_p = []
        for cit in range(len(data_set[2][2][p])):
            if cit not in ran_del:
                new_p.append(data_set[2][2][p][cit])

        data_set[2][2][p] = new_p

    return robust_data_set


def main():
    with open(get_file_base() + 'd2v_data/d2v_unstemmed.pickle', 'rb') as f:
        data_set = pickle.load(f)

    robust_data_set = generate_robust_ds(data_set)

    ds_ct = 0
    executor = ThreadPoolExecutor(max_workers=64)
    completed_robust_vecs = {}

    while ds_ct < len(data_set):
        p = 0
        while p < len(data_set[ds_ct][0]):
            futures = executor.submit(do_cosine, robust_data_set, ds_ct, p)

            completed_robust_vecs[data_set[ds_ct][3] + str(p)] = futures.result()
            p += 1
        ds_ct += 1

    write_to_file(get_file_base() + 'extracted_features/robustness.csv', completed_robust_vecs)


if __name__ == '__main__':
    main()
