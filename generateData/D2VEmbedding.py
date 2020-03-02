import json
import re
from gensim.models.doc2vec import Doc2Vec
import pickle
from general.baseFileExtractor import get_file_base, get_seminal_u, get_survey_u, get_uninfluential_u


def task_d2v(survey_hlp, seminal_hlp, uninfluential_hlp):
    # model 0 : dbow; model 1: dm
    doc2vec_model = Doc2Vec.load(get_file_base() + 'd2v_data/doc2vec_model_1')

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
        doc2vec_model.random.seed(0)
        hlp = doc2vec_model.infer_vector(p['abs'].split(), alpha=0.025, steps=20)
        uninfluential_p[ct] = hlp.reshape(1, -1)

        uninfluential_x[ct] = {}
        uninfluential_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            doc2vec_model.random.seed(0)
            hlp = doc2vec_model.infer_vector(ref['abs'].split(), alpha=0.025, steps=20)
            uninfluential_x[ct][ct_x] = hlp.reshape(1, -1)
            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            doc2vec_model.random.seed(0)
            hlp = doc2vec_model.infer_vector(cit['abs'].split(), alpha=0.025, steps=20)
            uninfluential_y[ct][ct_y] = hlp.reshape(1, -1)
            ct_y += 1

        ct += 1

    return [[seminal_p, seminal_x, seminal_y, 'sem '], [survey_p, survey_x, survey_y, 'surv '],
            [uninfluential_p, uninfluential_x, uninfluential_y, 'uni ']]


def make_d2v():
    # read in
    with open(get_survey_u(), encoding='latin-1') as s:
        survey_hlp = json.load(s)
    with open(get_seminal_u(), encoding='latin-1') as s:
        seminal_hlp = json.load(s)
    with open(get_uninfluential_u(), encoding='latin-1') as s:
        uninfluential_hlp = json.load(s)

    data_set = task_d2v(survey_hlp, seminal_hlp, uninfluential_hlp)

    with open(get_file_base() + 'd2v_data/d2v_unstemmed.pickle', 'wb') as output:
        pickle.dump(data_set, output)
