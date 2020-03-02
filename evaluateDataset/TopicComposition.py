
import json
from gensim.models import ldamodel
from gensim.corpora.dictionary import Dictionary
from general.baseFileExtractor import get_file_base, get_seminal_u, get_survey_u, get_uninfluential_u

# read in
with open(get_survey_u(), encoding='latin-1') as s:
    survey_hlp = json.load(s)
    survey_hlp = survey_hlp['survey']

with open(get_seminal_u(), encoding='latin-1') as s:
    seminal_hlp = json.load(s)
    seminal_hlp = seminal_hlp['seminal']

with open(get_uninfluential_u(), encoding='latin-1') as s:
    uninfluential_hlp = json.load(s)
    uninfluential_hlp = uninfluential_hlp['uninfluential']

lda = ldamodel.LdaModel.load(get_file_base() + 'lda_data/lda_model_unstemmed')
dictionary = Dictionary.load_from_text(get_file_base() + 'lda_data/dict_unstemmed')

sem = []
sur = []
uni = []
for p in seminal_hlp:
    sem.append(lda[dictionary.doc2bow(p['abs'].split())])
for p in survey_hlp:
    sur.append(lda[dictionary.doc2bow(p['abs'].split())])
for p in uninfluential_hlp:
    uni.append(lda[dictionary.doc2bow(p['abs'].split())])

fin_sem = []
fin_sur = []
fin_uni = []

for t in range(0, len(sem[0])):
    fin_sem.append(0)
    fin_sur.append(0)
    fin_uni.append(0)

for x in sem:
    for t in range(0, len(x)):
        fin_sem[t] += x[t][1]

for x in sur:
    for t in range(0, len(x)):
        fin_sur[t] += x[t][1]

for x in uni:
    for t in range(0, len(x)):
        fin_uni[t] += x[t][1]

print(fin_sem)
print(fin_sur)
print(fin_uni)
