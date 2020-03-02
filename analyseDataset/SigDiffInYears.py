import json
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s

# read in
with open(get_survey_s(), encoding='latin-1') as s:
    survey_hlp = json.load(s)
with open(get_seminal_s(), encoding='latin-1') as s:
    seminal_hlp = json.load(s)
with open(get_uninfluential_s(), encoding='latin-1') as s:
    uninfluential_hlp = json.load(s)

# years

sem_p = []
sur_p = []
uni_p = []
sem_ref = []
sur_ref = []
uni_ref = []
sem_cit = []
sur_cit = []
uni_cit = []

for p in seminal_hlp['seminal']:
    sem_p.append(p['year'])
    for ref in p['ref']:
        sem_ref.append(ref['year'])
    for cit in p['cit']:
        sem_cit.append(cit['year'])

for p in survey_hlp['survey']:
    sur_p.append(p['year'])
    for ref in p['ref']:
        sur_ref.append(ref['year'])
    for cit in p['cit']:
        sur_cit.append(cit['year'])

for p in uninfluential_hlp['uninfluential']:
    uni_p.append(p['year'])
    for ref in p['ref']:
        uni_ref.append(ref['year'])
    for cit in p['cit']:
        uni_cit.append(cit['year'])

# distances

sem_pref = []
sem_pcit = []
sem_refcit = []
sur_pref = []
sur_pcit = []
sur_refcit = []
uni_pref = []
uni_pcit = []
uni_refcit = []

for p in seminal_hlp['seminal']:
    for ref in p['ref']:
        sem_pref.append(abs(ref['year'] - p['year']))

        for cit in p['cit']:
            sem_refcit.append(abs(ref['year'] - cit['year']))
    for cit in p['cit']:
        sem_pcit.append(abs(cit['year'] - p['year']))

for p in seminal_hlp['seminal']:
    for ref in p['ref']:
        sur_pref.append(abs(ref['year'] - p['year']))

        for cit in p['cit']:
            sur_refcit.append(abs(ref['year'] - cit['year']))
    for cit in p['cit']:
        sur_pcit.append(abs(cit['year'] - p['year']))

for p in uninfluential_hlp['uninfluential']:
    for ref in p['ref']:
        uni_pref.append(abs(ref['year'] - p['year']))

        for cit in p['cit']:
            uni_refcit.append(abs(ref['year'] - cit['year']))
    for cit in p['cit']:
        uni_pcit.append(abs(cit['year'] - p['year']))

# print values for significance calculactions -> SPSS
for x in sem_ref:
    print(str(x) + " " + str(1))
for x in sur_ref:
    print(str(x) + " " + str(2))
for x in uni_ref:
    print(str(x) + " " + str(3))
