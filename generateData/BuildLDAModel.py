from gensim.models import ldamodel
from gensim.corpora.dictionary import Dictionary
from nltk.stem import PorterStemmer
from gensim.test.utils import datapath
from general.baseFileExtractor import get_lda_base, get_file_base


# build lda model from data specified in path_lda_base
def build_lda_model(stem):
    corpus = []
    ps = PorterStemmer()
    number_of_topics = 100

    # read in data from publications
    with open(get_lda_base(), 'r') as f:
        for line in f:
            if stem:
                stemmed = []

                for w in line.split():
                    s = ps.stem(w)
                    if len(s) > 1:
                        stemmed.append(s)

                corpus.append(stemmed)
            else:
                corpus.append(line.split())

    # build vocabulary and transform texts in vocab format
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]

    # do lda
    lda = ldamodel.LdaModel(corpus=corpus, num_topics=number_of_topics, passes=20, id2word=dictionary,
                            minimum_probability=0)

    if stem:
        temp_file = datapath('lda_model_stemmed')
        dictionary.save_as_text(get_file_base() + 'lda_data/dict_stemmed')
    else:
        temp_file = datapath('lda_model_unstemmed')
        dictionary.save_as_text(get_file_base() + 'lda_data/dict_unstemmed')

    lda.save(temp_file)
