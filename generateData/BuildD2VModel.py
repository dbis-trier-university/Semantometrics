from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from gensim.test.utils import get_tmpfile
from general.baseFileExtractor import get_d2v_base


# thanks to https://markroxor.github.io/gensim/static/notebooks/doc2vec-wikipedia.html
# thanks to https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb


class TaggedWikiDocument(object):

    def __init__(self, twiki):
        self.wiki = twiki
        self.wiki.metadata = True

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument(content, [title])


def build_d2v_model():
    result_file = open('res.txt', 'w+')
    result_file.write('start \n')
    result_file.flush()

    # include wikipedia dataset
    wiki = WikiCorpus(get_d2v_base())

    result_file.write('tag docs ready \n')

    result_file.flush()

    documents = TaggedWikiDocument(wiki)

    result_file.write('docs = wiki')
    result_file.flush()

    cores = multiprocessing.cpu_count()

    models = [
        # PV-DBOW
        Doc2Vec(dm=0, window=10, dbow_words=1, vector_size=300, min_count=20, epochs=20, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, window=10, dm_mean=1, vector_size=300, min_count=20, epochs=20, workers=cores),
    ]

    models[0].build_vocab(documents)
    models[1].reset_from(models[0])

    result_file.write('vocabulary built')
    result_file.flush()

    for model in models:
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
		
    models[0].save('doc2vec_model_0')
    models[1].save('doc2vec_model_1')