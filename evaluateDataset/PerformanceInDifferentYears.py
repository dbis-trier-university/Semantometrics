import json
from concurrent.futures import ThreadPoolExecutor
from computeFeatures.FeaturesFromEmbedding import do_cosine, write_to_file
from gensim.models.doc2vec import Doc2Vec
from general.baseFileExtractor import get_file_base, get_seminal_u, get_survey_u, get_uninfluential_u


def task_year(survey_hlp, seminal_hlp, uninfluential_hlp):
    # model 0 : dbow; model 1: dm
    doc2vec_model = Doc2Vec.load(get_file_base() + 'models/doc2vec_model_1')

    seminal_p = {}
    seminal_x = {}
    seminal_y = {}

    ct = 0
    curr_ct = 0
    for p in seminal_hlp['seminal']:
        if less_than_or_more == 'l':
            if p['year'] <= year:
                doc2vec_model.random.seed(0)
                hlp = doc2vec_model.infer_vector(p['abs'].split(), alpha=0.025, steps=20)
                seminal_p[curr_ct] = hlp.reshape(1, -1)

                seminal_x[curr_ct] = {}
                seminal_y[curr_ct] = {}

                ct_x = 0
                curr_ct_x = 0
                for ref in p['ref']:
                    if ref['year'] <= year:
                        doc2vec_model.random.seed(0)
                        hlp = doc2vec_model.infer_vector(ref['abs'].split(), alpha=0.025, steps=20)
                        seminal_x[curr_ct][curr_ct_x] = hlp.reshape(1, -1)
                        curr_ct_x += 1
                    ct_x += 1

                ct_y = 0
                curr_ct_y = 0
                for cit in p['cit']:
                    if cit['year'] <= year:
                        doc2vec_model.random.seed(0)
                        hlp = doc2vec_model.infer_vector(cit['abs'].split(), alpha=0.025, steps=20)
                        seminal_y[curr_ct][curr_ct_y] = hlp.reshape(1, -1)
                        curr_ct_y += 1
                    ct_y += 1

                curr_ct += 1
        if less_than_or_more == "m":
            if p['year'] >= year:
                doc2vec_model.random.seed(0)
                hlp = doc2vec_model.infer_vector(p['abs'].split(), alpha=0.025, steps=20)
                seminal_p[curr_ct] = hlp.reshape(1, -1)

                seminal_x[curr_ct] = {}
                seminal_y[curr_ct] = {}

                ct_x = 0
                curr_ct_x = 0
                for ref in p['ref']:
                    doc2vec_model.random.seed(0)
                    hlp = doc2vec_model.infer_vector(ref['abs'].split(), alpha=0.025, steps=20)
                    seminal_x[curr_ct][curr_ct_x] = hlp.reshape(1, -1)
                    curr_ct_x += 1
                    ct_x += 1

                ct_y = 0
                curr_ct_y = 0
                for cit in p['cit']:
                    doc2vec_model.random.seed(0)
                    hlp = doc2vec_model.infer_vector(cit['abs'].split(), alpha=0.025, steps=20)
                    seminal_y[curr_ct][curr_ct_y] = hlp.reshape(1, -1)
                    curr_ct_y += 1
                    ct_y += 1

                curr_ct += 1

        ct += 1

    survey_p = {}
    survey_x = {}
    survey_y = {}

    ct = 0
    curr_ct = 0
    for p in survey_hlp['survey']:
        if less_than_or_more == 'l':
            if p['year'] <= year:
                doc2vec_model.random.seed(0)
                hlp = doc2vec_model.infer_vector(p['abs'].split(), alpha=0.025, steps=20)
                survey_p[curr_ct] = hlp.reshape(1, -1)

                survey_x[curr_ct] = {}
                survey_y[curr_ct] = {}

                ct_x = 0
                curr_ct_x = 0
                for ref in p['ref']:
                    if ref['year'] <= year:
                        doc2vec_model.random.seed(0)
                        hlp = doc2vec_model.infer_vector(ref['abs'].split(), alpha=0.025, steps=20)
                        survey_x[curr_ct][curr_ct_x] = hlp.reshape(1, -1)
                        curr_ct_x += 1
                    ct_x += 1

                ct_y = 0
                curr_ct_y = 0
                for cit in p['cit']:
                    if cit['year'] <= year:
                        doc2vec_model.random.seed(0)
                        hlp = doc2vec_model.infer_vector(cit['abs'].split(), alpha=0.025, steps=20)
                        survey_y[curr_ct][curr_ct_y] = hlp.reshape(1, -1)
                        curr_ct_y += 1
                    ct_y += 1

                curr_ct += 1
        if less_than_or_more == 'm':
            if p['year'] >= year:
                doc2vec_model.random.seed(0)
                hlp = doc2vec_model.infer_vector(p['abs'].split(), alpha=0.025, steps=20)
                survey_p[curr_ct] = hlp.reshape(1, -1)

                survey_x[curr_ct] = {}
                survey_y[curr_ct] = {}

                ct_x = 0
                curr_ct_x = 0
                for ref in p['ref']:
                    doc2vec_model.random.seed(0)
                    hlp = doc2vec_model.infer_vector(ref['abs'].split(), alpha=0.025, steps=20)
                    survey_x[curr_ct][curr_ct_x] = hlp.reshape(1, -1)
                    curr_ct_x += 1
                    ct_x += 1

                ct_y = 0
                curr_ct_y = 0
                for cit in p['cit']:
                    doc2vec_model.random.seed(0)
                    hlp = doc2vec_model.infer_vector(cit['abs'].split(), alpha=0.025, steps=20)
                    survey_y[curr_ct][curr_ct_y] = hlp.reshape(1, -1)
                    curr_ct_y += 1
                    ct_y += 1

                curr_ct += 1
        ct += 1

    uninfluential_p = {}
    uninfluential_x = {}
    uninfluential_y = {}

    ct = 0
    curr_ct = 0
    for p in uninfluential_hlp['uninfluential']:
        if less_than_or_more == 'l':
            if p['year'] <= year:
                doc2vec_model.random.seed(0)
                hlp = doc2vec_model.infer_vector(p['abs'].split(), alpha=0.025, steps=20)
                uninfluential_p[curr_ct] = hlp.reshape(1, -1)

                uninfluential_x[curr_ct] = {}
                uninfluential_y[curr_ct] = {}

                ct_x = 0
                curr_ct_x = 0
                for ref in p['ref']:
                    if ref['year'] <= year:
                        doc2vec_model.random.seed(0)
                        hlp = doc2vec_model.infer_vector(ref['abs'].split(), alpha=0.025, steps=20)
                        uninfluential_x[curr_ct][curr_ct_x] = hlp.reshape(1, -1)
                        curr_ct_x += 1
                    ct_x += 1

                ct_y = 0
                curr_ct_y = 0
                for cit in p['cit']:
                    if cit['year'] <= year:
                        doc2vec_model.random.seed(0)
                        hlp = doc2vec_model.infer_vector(cit['abs'].split(), alpha=0.025, steps=20)
                        uninfluential_y[curr_ct][curr_ct_y] = hlp.reshape(1, -1)
                        curr_ct_y += 1
                    ct_y += 1

                curr_ct += 1
        if less_than_or_more == 'm':
            if p['year'] >= year:
                doc2vec_model.random.seed(0)
                hlp = doc2vec_model.infer_vector(p['abs'].split(), alpha=0.025, steps=20)
                uninfluential_p[curr_ct] = hlp.reshape(1, -1)

                uninfluential_x[curr_ct] = {}
                uninfluential_y[curr_ct] = {}

                ct_x = 0
                curr_ct_x = 0
                for ref in p['ref']:
                    doc2vec_model.random.seed(0)
                    hlp = doc2vec_model.infer_vector(ref['abs'].split(), alpha=0.025, steps=20)
                    uninfluential_x[curr_ct][curr_ct_x] = hlp.reshape(1, -1)
                    curr_ct_x += 1
                    ct_x += 1

                ct_y = 0
                curr_ct_y = 0
                for cit in p['cit']:
                    doc2vec_model.random.seed(0)
                    hlp = doc2vec_model.infer_vector(cit['abs'].split(), alpha=0.025, steps=20)
                    uninfluential_y[curr_ct][curr_ct_y] = hlp.reshape(1, -1)
                    curr_ct_y += 1
                    ct_y += 1

                curr_ct += 1
        ct += 1

    return [[seminal_p, seminal_x, seminal_y, 'sem '], [survey_p, survey_x, survey_y, 'surv '],
                                 [uninfluential_p, uninfluential_x, uninfluential_y, 'uni ']]


year = 2010
less_than_or_more = "m"


def main():
    with open(get_survey_u(), encoding='latin-1') as s:
        survey_hlp = json.load(s)
    with open(get_seminal_u(), encoding='latin-1') as s:
        seminal_hlp = json.load(s)
    with open(get_uninfluential_u(), encoding='latin-1') as s:
        uninfluential_hlp = json.load(s)

    data_set = task_year(survey_hlp, seminal_hlp, uninfluential_hlp)

    # calculate features on which the classification is going to be performed

    ds_ct = 0
    executor = ThreadPoolExecutor(max_workers=64)
    completed_vecs = {}

    # data_set[][0] -> P
    # data_set[][1] -> X, references
    # data_set[][2] -> Y, citations
    while ds_ct < len(data_set):
        p = 0
        while p < len(data_set[ds_ct][0]):
            futures = executor.submit(do_cosine, data_set, ds_ct, p)

            completed_vecs[data_set[ds_ct][3] + str(p)] = futures.result()
            p += 1
        ds_ct += 1

    write_to_file(get_file_base() + 'extracted_features/EVAL/d2v_cos_YEAR_' + less_than_or_more + "_" + str(year) +
                  '_unstemmed.csv', completed_vecs)


if __name__ == '__main__':
    main()
