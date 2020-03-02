import json
from concurrent.futures import ThreadPoolExecutor
import ast
from sklearn.externals import joblib
import pickle
from computeFeatures.FeaturesFromEmbedding import task_bert, task_lda
from computeFeatures.DistancesInYearsAsFeature import task_year
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s, get_stem, get_file_base


def do_one_doc_rep(data_set, ds_ct, p):
    one_doc_rep = [ds_ct]

    ref_dict = []

    for dim in range(0, len(data_set[ds_ct][0][0][0])):
        ref_dict.append(0)

    for ref in range(0, len(data_set[ds_ct][1][p])):
        for dim in range(0, len(data_set[ds_ct][1][p][ref][0])):
            ref_dict[dim] += data_set[ds_ct][1][p][ref][0][dim]

    for dim in range(0, len(data_set[ds_ct][1][0][0][0])):
        one_doc_rep.append(ref_dict[dim] / len(data_set[ds_ct][1][p]))

    # p
    for dim in range(0, len(data_set[ds_ct][0][0][0])):
        one_doc_rep.append(data_set[ds_ct][0][p][0][dim])

    cit_dict = []
    for dim in range(0, len(data_set[ds_ct][0][0][0])):
        cit_dict.append(0)

    for cit in range(0, len(data_set[ds_ct][2][p])):
        for dim in range(0, len(data_set[ds_ct][2][p][cit][0])):
            cit_dict[dim] += data_set[ds_ct][2][p][cit][0][dim]

    for dim in range(0, len(data_set[ds_ct][2][0][0][0])):
        one_doc_rep.append(cit_dict[dim] / len(data_set[ds_ct][2][p]))

    return one_doc_rep


def write_to_file(file_name, c_v):
    # write to file
    write_file = open(file_name, 'w+')

    first = True
    for t in range(0, len(c_v['sem 0']) - 1):
        if first:
            write_file.write('class')
        first = False
        write_file.write(',' + str(t))

    write_file.write('\n')

    for entry in c_v.keys():
        write_file.write(str(c_v[entry])[1:-1] + '\n')

    write_file.close()


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


# first argument: tfidf, d2v, bert, lda or years
task = 'lda'
# second argument: True or False
use_stemming = get_stem()


def main():
    if task not in ['tfidf', 'd2v', 'bert', 'lda', 'years']:
        print('Task ' + task + ' unknown.')
        return

    if task == 'tfidf':
        with open(get_file_base() + 'tfidf_data/tfidf_' + ('un' if not use_stemming else '') + 'stemmed.sav', 'rb') as \
                f:
            data_set = joblib.load(f)

        # todo: delete
        for p in range(0, len(data_set[2][0])):
            data_set[2][0][p] = [data_set[2][0][p]]

        for p in range(0, len(data_set[2][1])):
            for x in range(0, len(data_set[2][1][p])):
                data_set[2][1][p][x] = [data_set[2][1][p][x]]

        for p in range(0, len(data_set[2][2])):
            for x in range(0, len(data_set[2][2][p])):
                data_set[2][2][p][x] = [data_set[2][2][p][x]]

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
        data_set = task_lda(use_stemming)

    if task == 'years':
        with open(get_survey_s(), encoding='latin-1') as s:
            survey_hlp = json.load(s)
        with open(get_seminal_s(), encoding='latin-1') as s:
            seminal_hlp = json.load(s)
        with open(get_uninfluential_s(), encoding='latin-1') as s:
            uninfluential_hlp = json.load(s)

        data_set = task_year(survey_hlp, seminal_hlp, uninfluential_hlp)

        for ds in range(0, 3):
            for p in range(0, len(data_set[ds][0])):
                data_set[ds][0][p] = [[data_set[ds][0][p]]]

            for p in range(0, len(data_set[ds][1])):
                for x in range(0, len(data_set[ds][1][p])):
                    data_set[ds][1][p][x] = [[data_set[ds][1][p][x]]]

            for p in range(0, len(data_set[ds][2])):
                for x in range(0, len(data_set[ds][2][p])):
                    data_set[ds][2][p][x] = [[data_set[ds][2][p][x]]]

    ds_ct = 0
    executor = ThreadPoolExecutor(max_workers=64)
    completed_vecs = {}

    # data_set[][0] -> P
    # data_set[][1] -> X, references
    # data_set[][2] -> Y, citations
    while ds_ct < len(data_set):
        p = 0

        while p < len(data_set[ds_ct][0]):
            futures = executor.submit(do_one_doc_rep, data_set, ds_ct, p)

            completed_vecs[data_set[ds_ct][3] + str(p)] = futures.result()
            p += 1
        ds_ct += 1

    write_to_file(get_file_base() + 'extracted_features/OVR/' + task + '_' + ('un' if not use_stemming else '') +
                  'stemmed_OVR.csv', completed_vecs)


if __name__ == '__main__':
    main()
