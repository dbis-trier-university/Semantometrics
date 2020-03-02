from gensim.models import ldamodel
from gensim.corpora.dictionary import Dictionary
from nltk.stem import PorterStemmer
import json
import pandas
from general.baseFileExtractor import get_file_base, get_seminal_s, get_survey_s, get_uninfluential_s, \
    get_seminal_u, get_survey_u, get_uninfluential_u


# read in seminal, survey and uninfluential data
def read_in(file, p_type):
    raw = {}
    ref_p = {}
    cit_p = {}

    with open(file, 'r', encoding='utf8') as f:
        sem_j = json.load(f)
        for p in sem_j[p_type]:
            raw[p['key']] = p['abs'].split()

            ref_hlp = []
            for ref in p['ref']:
                ref_hlp.append(ref['abs'].split())

            cit_hlp = []
            for cit in p['cit']:
                cit_hlp.append(cit['abs'].split())

            ref_p[p['key']] = ref_hlp
            cit_p[p['key']] = cit_hlp
    return raw, ref_p, cit_p


# writes lda information of publications and associated references and citations to file
def transform_lda_topics_to_json(lda_topics):
    topics = {}

    for tuple_pairs in lda_topics:
        topics[tuple_pairs[0]] = tuple_pairs[1]

    return pandas.Series(topics).to_json(orient='values')


def write_to_file(file, file_one, raw, ref_p, cit_p, type_p, class_type, lda, dictionary):
    f = open(file, 'w', encoding='utf8')
    f_o = open(file_one, 'w', encoding='utf8')

    ct = 1
    pubs = []
    one_doc_reps = []
    for p in raw:
        # compute lda for seminal, survey and uninfluential data
        lda_p = transform_lda_topics_to_json(lda[dictionary.doc2bow(raw[p])])
        one_doc_rep = {'key': p, 'lda': lda_p}
        entry = {'key': p, 'lda': lda_p}

        references = ''
        citations = ''

        lda_ref = []
        lda_cit = []

        # compute lda for seminal, survey and uninfluential references and citations
        for a_r in ref_p[p]:
            lda_ref.append(transform_lda_topics_to_json(lda[dictionary.doc2bow(a_r)]))
            references += str(a_r) + ' '
        for a_c in cit_p[p]:
            lda_cit.append(transform_lda_topics_to_json(lda[dictionary.doc2bow(a_c)]))
            citations += str(a_c) + ' '

        entry['ref'] = lda_ref
        entry['cit'] = lda_cit

        one_doc_rep['ref'] = transform_lda_topics_to_json(lda[dictionary.doc2bow([references])])
        one_doc_rep['cit'] = transform_lda_topics_to_json(lda[dictionary.doc2bow([citations])])

        print(str(ct))
        pubs.append(entry)
        one_doc_reps.append(one_doc_rep)
        ct += 1

    f.write(json.dumps({type_p: pubs}) + '\n')
    f.close()

    f_o.write(json.dumps({type_p: one_doc_reps}) + '\n')
    f_o.close()


# build lda model from data of dblp xml file and corresponding aux data
def make_lda(stem):
    ps = PorterStemmer()

    if stem:
        lda = ldamodel.LdaModel.load(get_file_base() + 'lda_data/lda_model_stemmed')
        dictionary = Dictionary.load_from_text(get_file_base() + 'lda_data/dict_stemmed')

        sem_raw, sem_in, sem_out = read_in(get_seminal_s(), 'seminal')
        sur_raw, sur_in, sur_out = read_in(get_survey_s(), 'survey')
        uni_raw, uni_in, uni_out = read_in(get_uninfluential_s(), 'uninfluential')
    else:
        lda = ldamodel.LdaModel.load(get_file_base() + 'lda_data/lda_model_unstemmed')
        dictionary = Dictionary.load_from_text(get_file_base() + 'lda_data/dict_unstemmed')

        sem_raw, sem_in, sem_out = read_in(get_seminal_u(), 'seminal')
        sur_raw, sur_in, sur_out = read_in(get_survey_u(), 'survey')
        uni_raw, uni_in, uni_out = read_in(get_uninfluential_u(), 'uninfluential')

    # write lda information to file
    if stem:
        write_to_file(get_file_base() + 'lda_data/sem_lda_stemmed.json',
                      get_file_base() + 'lda_data/sem_lda_stemmed_one_doc_rep.json',
                      sem_raw, sem_in, sem_out, 'seminal', '0', lda, dictionary)

        write_to_file(get_file_base() + 'lda_data/sur_lda_stemmed.json',
                      get_file_base() + 'lda_data/sur_lda_stemmed_one_doc_rep.json',
                      sur_raw, sur_in, sur_out, 'survey', '1', lda, dictionary)

        write_to_file(get_file_base() + 'lda_data/uni_lda_stemmed.json',
                      get_file_base() + 'lda_data/uni_lda_stemmed_one_doc_rep.json',
                      uni_raw, uni_in, uni_out, 'uninfluential', '2', lda, dictionary)
    else:
        write_to_file(get_file_base() + 'lda_data/sem_lda_unstemmed.json',
                      get_file_base() + 'lda_data/sem_lda_unstemmed_one_doc_rep.json',
                      sem_raw, sem_in, sem_out, 'seminal', '0', lda, dictionary)

        write_to_file(get_file_base() + 'lda_data/sur_lda_unstemmed.json',
                      get_file_base() + 'lda_data/sur_lda_unstemmed_one_doc_rep.json',
                      sur_raw, sur_in, sur_out, 'survey', '1', lda, dictionary)

        write_to_file(get_file_base() + 'lda_data/uni_lda_unstemmed.json',
                      get_file_base() + 'lda_data/uni_lda_unstemmed_one_doc_rep.json',
                      uni_raw, uni_in, uni_out, 'uninfluential', '2', lda, dictionary)
