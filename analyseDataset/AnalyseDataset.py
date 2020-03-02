import json
import numpy as np
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s, get_seminal_u,\
    get_survey_u, get_uninfluential_u, get_stem

use_stemming = get_stem()

if use_stemming:
    with open(get_seminal_s(), 'r', encoding='utf8') as f:
        sem = json.load(f)
        sem = sem['seminal']
    with open(get_survey_s(), 'r', encoding='utf8') as f:
        sur = json.load(f)
        sur = sur['survey']
    with open(get_uninfluential_s(), 'r', encoding='utf8') as f:
        uni = json.load(f)
        uni = uni['uninfluential']
else:
    with open(get_seminal_u(), 'r', encoding='utf8') as f:
        sem = json.load(f)
        sem = sem['seminal']
    with open(get_survey_u(), 'r', encoding='utf8') as f:
        sur = json.load(f)
        sur = sur['survey']
    with open(get_uninfluential_u(), 'r', encoding='utf8') as f:
        uni = json.load(f)
        uni = uni['uninfluential']

avg_length_abs_sem = 0
ref_sem = []
cit_sem = []

for p in sem:
    avg_length_abs_sem += len(p['abs'].split())
    ref_sem.append(len(p['ref']))
    cit_sem.append(len(p['cit']))

avg_length_abs_sur = 0
ref_sur = []
cit_sur = []

for p in sur:
    avg_length_abs_sur += len(p['abs'].split())
    ref_sur.append(len(p['ref']))
    cit_sur.append(len(p['cit']))

avg_length_abs_uni = 0
ref_uni = []
cit_uni = []

for p in uni:
    avg_length_abs_uni += len(p['abs'].split())
    ref_uni.append(len(p['ref']))
    cit_uni.append(len(p['cit']))

print('sem')
print(avg_length_abs_sem/660)
print('_______________________')
print(np.sum(ref_sem))
print(np.mean(ref_sem))
print(np.min(ref_sem))
print(np.max(ref_sem))
print(np.percentile(ref_sem, 25))
print(np.percentile(ref_sem, 50))
print(np.percentile(ref_sem, 75))
print('_______________________')
print(np.sum(cit_sem))
print(np.mean(cit_sem))
print(np.min(cit_sem))
print(np.max(cit_sem))
print(np.percentile(cit_sem, 25))
print(np.percentile(cit_sem, 50))
print(np.percentile(cit_sem, 75))

print('_______________________')
print('sur')
print(avg_length_abs_sur/660)
print('_______________________')
print(np.sum(ref_sur))
print(np.mean(ref_sur))
print(np.min(ref_sur))
print(np.max(ref_sur))
print(np.percentile(ref_sur, 25))
print(np.percentile(ref_sur, 50))
print(np.percentile(ref_sur, 75))
print('_______________________')
print(np.sum(cit_sur))
print(np.mean(cit_sur))
print(np.min(cit_sur))
print(np.max(cit_sur))
print(np.percentile(cit_sur, 25))
print(np.percentile(cit_sur, 50))
print(np.percentile(cit_sur, 75))

print('_______________________')
print('uni')
print(avg_length_abs_uni/660)
print('_______________________')
print(np.sum(ref_uni))
print(np.mean(ref_uni))
print(np.min(ref_uni))
print(np.max(ref_uni))
print(np.percentile(ref_uni, 25))
print(np.percentile(ref_uni, 50))
print(np.percentile(ref_uni, 75))
print('_______________________')
print(np.sum(cit_uni))
print(np.mean(cit_uni))
print(np.min(cit_uni))
print(np.max(cit_uni))
print(np.percentile(cit_uni, 25))
print(np.percentile(cit_uni, 50))
print(np.percentile(cit_uni, 75))