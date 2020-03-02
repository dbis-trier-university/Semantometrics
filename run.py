import json
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s, get_seminal_u, get_survey_u, \
    get_uninfluential_u, get_stem, get_file_base, get_what_to_do, get_which_vectors, get_which_distance, get_classifier
from generateData.TFIDFEmbedding import make_tfidf
from generateData.BuildD2VModel import build_d2v_model
from generateData.D2VEmbedding import make_d2v
from generateData.BERTEmbedding import make_bert
from generateData.BuildLDAModel import build_lda_model
from generateData.LDAEmbedding import make_lda
from computeFeatures.FeaturesFromEmbedding import make_features
from classify.ClassificationSEMallC import sem_all_class
from classify.ClassificationSEM import sem_one_class

import os


# build folder structure
if not os.path.exists(os.path.dirname(get_file_base() + 'tfidf_data/')):
    os.makedirs(os.path.dirname(get_file_base() + 'tfidf_data/'))
if not os.path.exists(os.path.dirname(get_file_base() + 'd2v_data/')):
    os.makedirs(os.path.dirname(get_file_base() + 'd2v_data/'))
if not os.path.exists(os.path.dirname(get_file_base() + 'bert_data/')):
    os.makedirs(os.path.dirname(get_file_base() + 'bert_data/'))
if not os.path.exists(os.path.dirname(get_file_base() + 'lda_data/')):
    os.makedirs(os.path.dirname(get_file_base() + 'lda_data/'))

if not os.path.exists(os.path.dirname(get_file_base() + 'extracted_features/')):
    os.makedirs(os.path.dirname(get_file_base() + 'extracted_features/'))
if not os.path.exists(os.path.dirname(get_file_base() + 'extracted_features/OVR/')):
    os.makedirs(os.path.dirname(get_file_base() + 'extracted_features/OVR/'))
if not os.path.exists(os.path.dirname(get_file_base() + 'plots/')):
    os.makedirs(os.path.dirname(get_file_base() + 'plots/'))

# read in base data
with open(get_seminal_s(), encoding='latin-1') as s:
    seminal_hlp_s = json.load(s)
with open(get_survey_s(), encoding='latin-1') as s:
    survey_hlp_s = json.load(s)
with open(get_uninfluential_s(), encoding='latin-1') as s:
    uninfluential_hlp_s = json.load(s)

with open(get_seminal_u(), encoding='latin-1') as s:
    seminal_hlp_u = json.load(s)
with open(get_survey_u(), encoding='latin-1') as s:
    survey_hlp_u = json.load(s)
with open(get_uninfluential_u(), encoding='latin-1') as s:
    uninfluential_hlp_u = json.load(s)

# specification of what user wants to compute
if get_what_to_do() == 0:
    # creation of all vector representations

    # stemmed tf-idf vectors
    make_tfidf(True)
    # unstemmed tf-idf vectors
    make_tfidf(False)

    # doc2vec model and embedding
    build_d2v_model()
    make_d2v()

    # BERT embedding
    make_bert()

    # stemmed LDA model and embedding
    build_lda_model(True)
    make_lda(True)
    # unstemmed LDA model and embedding
    build_lda_model(False)
    make_lda(False)

if get_what_to_do() == 1:
    # creation of a single vector representation
    if get_which_vectors() == 'TFIDF-s':
        # stemmed tf-idf vectors
        make_tfidf(True)
    if get_which_vectors() == 'TFIDF-u':
        # unstemmed tf-idf vectors
        make_tfidf(False)

    if get_which_vectors() == 'D2V':
        # doc2vec model and embedding
        build_d2v_model()
        make_d2v()

    if get_which_vectors() == 'BERT':
        # BERT embedding
        make_bert()

    if get_which_vectors() == 'LDA-s':
        # stemmed LDA model and embedding
        build_lda_model(True)
        make_lda(True)
    if get_which_vectors() == 'LDA-u':
        # unstemmed LDA model and embedding
        build_lda_model(False)
        make_lda(False)

if get_what_to_do() == 2:
    # construct all features for semantometrics
    make_features('cos', 'tfidf', True)
    make_features('jac', 'tfidf', True)
    make_features('ipd', 'tfidf', True)
    make_features('cos', 'tfidf', False)
    make_features('jac', 'tfidf', False)
    make_features('ipd', 'tfidf', False)

    make_features('cos', 'd2v', False)
    make_features('jac', 'd2v', False)
    make_features('ipd', 'd2v', False)

    make_features('cos', 'bert', False)
    make_features('jac', 'bert', False)
    make_features('ipd', 'bert', False)

    make_features('emd', 'lda', True)
    make_features('ipd', 'lda', True)
    make_features('emd', 'lda', False)
    make_features('ipd', 'lda', False)

    make_features('dist', 'year', False)

if get_what_to_do() == 3:
    # construct one feature for semantometrics for a combination of vector and distance
    vec = get_which_vectors()
    dist = get_which_distance()
    if (vec in ['tfidf', 'd2v', 'bert'] and dist in ['cos', 'jac', 'ipd']) or (vec == 'lda' and dist in ['emd', 'ipd'])\
            or (vec == 'year' and dist == 'dist'):
        make_features(dist, vec, get_stem())

if get_what_to_do() == 4:
    # all classifier classifications on semantometrics for a combination of vector and distance

    vec = get_which_vectors()
    dist = get_which_distance()
    if (vec in ['tfidf', 'd2v', 'bert'] and dist in ['cos', 'jac', 'ipd']) or (vec == 'lda' and dist in ['emd', 'ipd'])\
            or (vec == 'year' and dist == 'dist'):
        sem_all_class(get_which_vectors(), get_which_distance(), get_stem(), False, False)

if get_what_to_do() == 5:
    # single classifier classifications on semantometrics for a combination of vector and distance

    vec = get_which_vectors()
    dist = get_which_distance()
    if (vec in ['tfidf', 'd2v', 'bert'] and dist in ['cos', 'jac', 'ipd']) or (vec == 'lda' and dist in ['emd', 'ipd'])\
            or (vec == 'year' and dist == 'dist'):
        sem_one_class(get_which_vectors(), get_which_distance(), get_classifier(), get_stem(), False, '', False)

if get_what_to_do() == 6:
    # all classifier classifications on semantometrics with information available at publication time for a combination
    # of vector and distance
    sem_all_class(get_which_vectors(), get_which_distance(), get_stem(), False, True)

if get_what_to_do() == 7:
    # single classifier classification on semantometrics with information available at publication time for a
    # combination of vector and distance
    sem_one_class(get_which_vectors(), get_which_distance(), get_classifier(), get_stem(), False, '', True)
