import json
from general.baseFileExtractor import get_seminal_u, get_survey_u, get_uninfluential_u

with open(get_seminal_u(), 'r', encoding='utf8') as f:
    sem = json.load(f)
    sem = sem['seminal']
with open(get_survey_u(), 'r', encoding='utf8') as f:
    sur = json.load(f)
    sur = sur['survey']
with open(get_uninfluential_u(), 'r', encoding='utf8') as f:
    uni = json.load(f)
    uni = uni['uninfluential']

sur_ct = 0
for p in sur:
    # iterate over references
    if len(p['ref']) >= 25:
        sur_ct += 1

print('# survey p with at least 25 references: ' + str(sur_ct))

sem_ct = 0
for p in sem:
    years = {}

    # iterate over citations
    for cit in p['cit']:
        y = cit['year']

        if y in years:
            years[y] = years[y] + 1
        else:
            years[y] = 1

    for y in years:
        if years[y] >= 4:
            sem_ct += 1
            break

print('# seminal p with at least 4 citations in a year: ' + str(sem_ct))

dist_c = 0
dist_r = 0
for p in sem:
    # iterate over references
    curr_c = 0
    for cit in p['cit']:
        curr_c += abs(cit['year'] - p['year'])

    if len(p['cit']) > 0:
        dist_c += (curr_c/len(p['cit']))

    curr_r = 0
    for ref in p['ref']:
        curr_r += abs(p['year'] - ref['year'])

    dist_r += (curr_r / len(p['ref']))

print('sem ' + str(dist_r/660) + ' ' + str(dist_c/660))

dist_c = 0
dist_r = 0
for p in sur:
    # iterate over references
    curr_c = 0
    for cit in p['cit']:
        curr_c += abs(cit['year'] - p['year'])

    if len(p['cit']) > 0:
        dist_c += (curr_c/len(p['cit']))

    curr_r = 0
    for ref in p['ref']:
        curr_r += abs(p['year'] - ref['year'])

    dist_r += (curr_r / len(p['ref']))

print('sur ' + str(dist_r/660) + ' ' + str(dist_c/660))

dist_c = 0
dist_r = 0
for p in uni:
    # iterate over references
    curr_c = 0
    for cit in p['cit']:
        curr_c += abs(cit['year'] - p['year'])

    if len(p['cit']) > 0:
        dist_c += (curr_c/len(p['cit']))

    curr_r = 0
    for ref in p['ref']:
        curr_r += abs(p['year'] - ref['year'])

    dist_r += (curr_r / len(p['ref']))

print('uni ' + str(dist_r/660) + ' ' + str(dist_c/660))

uni_ct = 0
for p in uni:
    # iterate over references
    if len(p['ref']) >= 25:
        uni_ct += 1

print('# uninfluential p with at least 25 references: ' + str(uni_ct))

uni_ct = 0
for p in uni:
    years = {}

    # iterate over citations
    for cit in p['cit']:
        y = cit['year']

        if y in years:
            years[y] = years[y] + 1
        else:
            years[y] = 1

    for y in years:
        if years[y] >= 4:
            uni_ct += 1
            break

print('# uninfluential p with at least 4 citations in a year: ' + str(uni_ct))
