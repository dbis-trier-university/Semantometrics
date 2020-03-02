import json
import pandas as pd
from classify.ClassificationSEM import all_single_feature_classify_data, random_state
from general.baseFileExtractor import get_seminal_s, get_survey_s, get_uninfluential_s


def main():
    with open(get_seminal_s(), 'r', encoding='utf8') as f:
        sem = json.load(f)
        sem = sem['seminal']
    with open(get_survey_s(), 'r', encoding='utf8') as f:
        sur = json.load(f)
        sur = sur['survey']
    with open(get_uninfluential_s(), 'r', encoding='utf8') as f:
        uni = json.load(f)
        uni = uni['uninfluential']

    references = []
    citations = []
    labels = []

    for p in sem:
        references.append(len(p['ref']))
        citations.append(len(p['cit']))
        labels.append(0)

    for p in sur:
        references.append(len(p['ref']))
        citations.append(len(p['cit']))
        labels.append(1)

    for p in uni:
        references.append(len(p['ref']))
        citations.append(len(p['cit']))
        labels.append(2)

    data = pd.DataFrame(data={'class': labels, 'r': references, 'c': citations})
    data = data.sample(frac=1, random_state=random_state)

    labels = data[['class']]
    data.drop(['class'], axis=1, inplace=True)

    all_single_feature_classify_data(data, labels, 'GB')


if __name__ == "__main__":
    main()
