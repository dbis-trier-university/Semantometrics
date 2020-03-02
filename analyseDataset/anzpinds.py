import json
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s


def count(survey_hlp, seminal_hlp, uninfluential_hlp):
    publication_keys = set()

    for p in seminal_hlp['seminal']:

        if p['key'] not in publication_keys:
            publication_keys.add(p['key'])

        for ref in p['ref']:
            if ref['key'] not in publication_keys:
                publication_keys.add(ref['key'])

        for cit in p['cit']:
            if cit['key'] not in publication_keys:
                publication_keys.add(cit['key'])

    for p in survey_hlp['survey']:

        if p['key'] not in publication_keys:
            publication_keys.add(p['key'])

        for ref in p['ref']:
            if ref['key'] not in publication_keys:
                publication_keys.add(ref['key'])

        for cit in p['cit']:
            if cit['key'] not in publication_keys:
                publication_keys.add(cit['key'])

    for p in uninfluential_hlp['uninfluential']:

        if p['key'] not in publication_keys:
            publication_keys.add(p['key'])

        for ref in p['ref']:
            if ref['key'] not in publication_keys:
                publication_keys.add(ref['key'])

        for cit in p['cit']:
            if cit['key'] not in publication_keys:
                publication_keys.add(cit['key'])

    print(str(len(publication_keys)))


def main():
    # read in
    with open(get_survey_s(), encoding='latin-1') as s:
        survey_hlp = json.load(s)
    with open(get_seminal_s(), encoding='latin-1') as s:
        seminal_hlp = json.load(s)
    with open(get_uninfluential_s(), encoding='latin-1') as s:
        uninfluential_hlp = json.load(s)

    count(survey_hlp, seminal_hlp, uninfluential_hlp)


if __name__ == '__main__':
    main()
