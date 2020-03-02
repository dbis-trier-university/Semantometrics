import json
import numpy as np
from sklearn.externals import joblib
from general.baseFileExtractor import get_file_base, get_seminal_s, get_survey_s, get_uninfluential_s, \
    get_seminal_u, get_survey_u, get_uninfluential_u


# TF IDF
def tf_idf(raw_data, words, idf, use_stemming):
    data = raw_data.split()

    vec = np.zeros(len(words))
    for dw in data:
        vec[words[dw]] = vec[words[dw]] + 1

    d_sum = 0

    # tf idf
    d_i = 0
    while d_i < len(words):
        vec[d_i] = vec[d_i] * idf[d_i]
        d_sum += vec[d_i]
        d_i += 1

    # length normalization
    d_i = 0
    while d_i < len(words):
        vec[d_i] = vec[d_i] / d_sum
        d_i += 1

    data = vec.reshape(1, -1)

    return data


def task_tfidf(survey_hlp, seminal_hlp, uninfluential_hlp, use_stemming):
    # preparation -> count
    doc_freq = {}
    words = {}

    publication_keys = set()

    if use_stemming:
        forbidden_words = ['review', 'survei']
    else:
        forbidden_words = ['review', 'survey']

    seminal_p = {}
    seminal_x = {}
    seminal_y = {}

    ct = 0
    for p in seminal_hlp['seminal']:
        words_in_p = set()
        sem_abs = ''
        for w in p['abs'].split():
            # ignore word if it is either survey or review

            if drop_sur_rev and w in forbidden_words:
                if w not in words_in_p:
                    words_in_p.add(w)
            else:
                if w not in words:
                    words[w] = len(words)

                sem_abs += w + ' '
                if w not in words_in_p:
                    words_in_p.add(w)

        if p['key'] not in publication_keys:
            publication_keys.add(p['key'])
            # put word of p in docfreq
            for w in words_in_p:
                if w in doc_freq:
                    doc_freq[w] = doc_freq[w] + 1
                else:
                    doc_freq[w] = 1

        seminal_p[ct] = sem_abs

        seminal_x[ct] = {}
        seminal_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            words_in_p = set()
            abs_ref = ''
            for w in ref['abs'].split():
                # ignore word if it is either survey or review
                if drop_sur_rev and w in forbidden_words:
                    if w not in words_in_p:
                        words_in_p.add(w)
                else:
                    if w not in words:
                        words[w] = len(words)

                    abs_ref += w + ' '
                    if w not in words_in_p:
                        words_in_p.add(w)

            seminal_x[ct][ct_x] = abs_ref

            # put word of p in docfreq
            if ref['key'] not in publication_keys:
                publication_keys.add(ref['key'])
                for w in words_in_p:
                    if w in doc_freq:
                        doc_freq[w] = doc_freq[w] + 1
                    else:
                        doc_freq[w] = 1

            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            words_in_p = set()
            abs_cit = ''
            for w in cit['abs'].split():
                # ignore word if it is either survey or review
                if drop_sur_rev and w in forbidden_words:
                    if w not in words_in_p:
                        words_in_p.add(w)
                else:
                    if w not in words:
                        length = len(words)
                        words[w] = length

                    abs_cit += w + ' '
                    if w not in words_in_p:
                        words_in_p.add(w)

            seminal_y[ct][ct_y] = abs_cit

            # put word of p in docfreq
            if cit['key'] not in publication_keys:
                publication_keys.add(cit['key'])
                for w in words_in_p:
                    if w in doc_freq:
                        doc_freq[w] = doc_freq[w] + 1
                    else:
                        doc_freq[w] = 1

            ct_y += 1

        ct += 1

    survey_p = {}
    survey_x = {}
    survey_y = {}

    ct = 0
    for p in survey_hlp['survey']:
        words_in_p = set()
        sem_abs = ''
        for w in p['abs'].split():
            # ignore word if it is either survey or review
            if drop_sur_rev and w in forbidden_words:
                if w not in words_in_p:
                    words_in_p.add(w)
            else:
                if w not in words:
                    length = len(words)
                    words[w] = length

                sem_abs += w + ' '
                if w not in words_in_p:
                    words_in_p.add(w)

        # put word of p in docfreq
        if p['key'] not in publication_keys:
            publication_keys.add(p['key'])
            for w in words_in_p:
                if w in doc_freq:
                    doc_freq[w] = doc_freq[w] + 1
                else:
                    doc_freq[w] = 1

        survey_p[ct] = sem_abs

        survey_x[ct] = {}
        survey_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            words_in_p = set()
            abs_ref = ''
            for w in ref['abs'].split():
                # ignore word if it is either survey or review
                if drop_sur_rev and w in forbidden_words:
                    if w not in words_in_p:
                        words_in_p.add(w)
                else:
                    if w not in words:
                        length = len(words)
                        words[w] = length

                    abs_ref += w + ' '
                    if w not in words_in_p:
                        words_in_p.add(w)

            survey_x[ct][ct_x] = abs_ref

            # put word of p in docfreq
            if ref['key'] not in publication_keys:
                publication_keys.add(ref['key'])
                for w in words_in_p:
                    if w in doc_freq:
                        doc_freq[w] = doc_freq[w] + 1
                    else:
                        doc_freq[w] = 1

            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            words_in_p = set()
            abs_cit = ''
            for w in cit['abs'].split():
                # ignore word if it is either survey or review
                if drop_sur_rev and w in forbidden_words:
                    if w not in words_in_p:
                        words_in_p.add(w)
                else:
                    if w not in words:
                        length = len(words)
                        words[w] = length

                    abs_cit += w + ' '
                    if w not in words_in_p:
                        words_in_p.add(w)

            survey_y[ct][ct_y] = abs_cit

            # put word of p in document frequency
            if cit['key'] not in publication_keys:
                publication_keys.add(cit['key'])
                for w in words_in_p:
                    if w in doc_freq:
                        doc_freq[w] = doc_freq[w] + 1
                    else:
                        doc_freq[w] = 1

            ct_y += 1
        ct += 1

    uninfluential_p = {}
    uninfluential_x = {}
    uninfluential_y = {}

    ct = 0
    for p in uninfluential_hlp['uninfluential']:
        words_in_p = set()
        sem_abs = ''
        for w in p['abs'].split():
            # ignore word if it is either survey or review
            if drop_sur_rev and w in forbidden_words:
                if w not in words_in_p:
                    words_in_p.add(w)
            else:
                if w not in words:
                    words[w] = len(words)

                sem_abs += w + ' '
                if w not in words_in_p:
                    words_in_p.add(w)

        if p['key'] not in publication_keys:
            publication_keys.add(p['key'])
            # put word of p in docfreq
            for w in words_in_p:
                if w in doc_freq:
                    doc_freq[w] = doc_freq[w] + 1
                else:
                    doc_freq[w] = 1

        uninfluential_p[ct] = sem_abs

        uninfluential_x[ct] = {}
        uninfluential_y[ct] = {}

        ct_x = 0
        for ref in p['ref']:
            words_in_p = set()
            abs_ref = ''
            for w in ref['abs'].split():
                # ignore word if it is either survey or review
                if drop_sur_rev and w in forbidden_words:
                    if w not in words_in_p:
                        words_in_p.add(w)
                else:
                    if w not in words:
                        words[w] = len(words)

                    abs_ref += w + ' '
                    if w not in words_in_p:
                        words_in_p.add(w)

            uninfluential_x[ct][ct_x] = abs_ref

            # put word of p in docfreq
            if ref['key'] not in publication_keys:
                publication_keys.add(ref['key'])
                for w in words_in_p:
                    if w in doc_freq:
                        doc_freq[w] = doc_freq[w] + 1
                    else:
                        doc_freq[w] = 1

            ct_x += 1

        ct_y = 0
        for cit in p['cit']:
            words_in_p = set()
            abs_cit = ''
            for w in cit['abs'].split():
                # ignore word if it is either survey or review
                if drop_sur_rev and w in forbidden_words:
                    if w not in words_in_p:
                        words_in_p.add(w)
                else:
                    if w not in words:
                        length = len(words)
                        words[w] = length

                    abs_cit += w + ' '
                    if w not in words_in_p:
                        words_in_p.add(w)

            uninfluential_y[ct][ct_y] = abs_cit

            # put word of p in docfreq
            if cit['key'] not in publication_keys:
                publication_keys.add(cit['key'])
                for w in words_in_p:
                    if w in doc_freq:
                        doc_freq[w] = doc_freq[w] + 1
                    else:
                        doc_freq[w] = 1

            ct_y += 1

        ct += 1

    # ------------------------------------------------------------------------------------------------------------------
    idf = {}

    for w in doc_freq:
        idf[words[w]] = len(publication_keys) / doc_freq[w]

    # seminal
    i = 0
    while i < len(seminal_p):
        seminal_p[i] = tf_idf(seminal_p[i], words, idf, use_stemming)
        i += 1

    i = 0
    while i < len(seminal_x):
        j = 0
        while j < len(seminal_x[i]):
            seminal_x[i][j] = tf_idf(seminal_x[i][j], words, idf, use_stemming)
            j += 1
        i += 1

    i = 0
    while i < len(seminal_y):
        j = 0
        while j < len(seminal_y[i]):
            seminal_y[i][j] = tf_idf(seminal_y[i][j], words, idf, use_stemming)
            j += 1
        i += 1

    # survey
    i = 0
    while i < len(survey_p):
        survey_p[i] = tf_idf(survey_p[i], words, idf, use_stemming)
        i += 1

    i = 0
    while i < len(survey_x):
        j = 0
        while j < len(survey_x[i]):
            survey_x[i][j] = tf_idf(survey_x[i][j], words, idf, use_stemming)
            j += 1
        i += 1

    i = 0
    while i < len(survey_y):
        j = 0
        while j < len(survey_y[i]):
            survey_y[i][j] = tf_idf(survey_y[i][j], words, idf, use_stemming)
            j += 1
        i += 1

    # uninfluential
    i = 0
    while i < len(uninfluential_p):
        uninfluential_p[i] = tf_idf(uninfluential_p[i], words, idf, use_stemming)
        i += 1

    i = 0
    while i < len(uninfluential_x):
        j = 0
        while j < len(uninfluential_x[i]):
            uninfluential_x[i][j] = tf_idf(uninfluential_x[i][j], words, idf, use_stemming)
            j += 1
        i += 1

    i = 0
    while i < len(uninfluential_y):
        j = 0
        while j < len(uninfluential_y[i]):
            uninfluential_y[i][j] = tf_idf(uninfluential_y[i][j], words, idf, use_stemming)
            j += 1
        i += 1

    return [[seminal_p, seminal_x, seminal_y, 'sem '], [survey_p, survey_x, survey_y, 'surv '],
            [uninfluential_p, uninfluential_x, uninfluential_y, 'uni ']], words


def print_words(words, use_stemming):
    if use_stemming:
        f = open(get_file_base() + 'tfidf_data/words_stemmed.txt', 'w', encoding='utf8')
    else:
        f = open(get_file_base() + 'tfidf_data/words_unstemmed.txt', 'w', encoding='utf8')

    for word, w_id in words.items():
        f.write(str(w_id) + ' ' + word + '\n')

    f.close()


drop_sur_rev = False


def make_tfidf(use_stemming):
    # read in
    if use_stemming:
        with open(get_survey_s(), encoding='latin-1') as s:
            survey_hlp = json.load(s)
        with open(get_seminal_s(), encoding='latin-1') as s:
            seminal_hlp = json.load(s)
        with open(get_uninfluential_s(), encoding='latin-1') as s:
            uninfluential_hlp = json.load(s)
    else:
        with open(get_survey_u(), encoding='latin-1') as s:
            survey_hlp = json.load(s)
        with open(get_seminal_u(), encoding='latin-1') as s:
            seminal_hlp = json.load(s)
        with open(get_uninfluential_u(), encoding='latin-1') as s:
            uninfluential_hlp = json.load(s)

    data_set, words = task_tfidf(survey_hlp, seminal_hlp, uninfluential_hlp, use_stemming)

    if use_stemming:
        with open(get_file_base() + 'tfidf_data/tfidf_stemmed.sav', 'wb') as output:
            joblib.dump(data_set, output)
    else:
        with open(get_file_base() + 'tfidf_data/tfidf_unstemmed.sav', 'wb') as output:
            joblib.dump(data_set, output)

    print_words(words, use_stemming)
