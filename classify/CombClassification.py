import pandas
from evaluateMethod.ClassificationOVR import all_single_feature_classify_data
from general.baseFileExtractor import get_file_base

random_state = 100


def read_in_csv_data_4(source_one, source_two, source_three, source_four):
    data_one = pandas.read_csv(source_one, sep=',')
    data_two = pandas.read_csv(source_two, sep=',')
    data_three = pandas.read_csv(source_three, sep=',')
    data_four = pandas.read_csv(source_four, sep=',')
    data_two.drop(['class'], axis=1, inplace=True)
    data_three.drop(['class'], axis=1, inplace=True)
    data_four.drop(['class'], axis=1, inplace=True)

    columns = data_one.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'one_' + str(columns[col])

    columns = data_two.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'two_' + str(columns[col])

    columns = data_three.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'three_' + str(columns[col])

    data = pandas.concat([data_one, data_two, data_three, data_four], axis=1, sort=False)

    data = data.sample(frac=1, random_state=random_state)

    labels = data[['class']]
    data.drop(['class'], axis=1, inplace=True)

    return data, labels


def read_in_csv_data_3(source_one, source_two, source_three):
    data_one = pandas.read_csv(source_one, sep=',')
    data_two = pandas.read_csv(source_two, sep=',')
    data_three = pandas.read_csv(source_three, sep=',')
    data_two.drop(['class'], axis=1, inplace=True)
    data_three.drop(['class'], axis=1, inplace=True)

    columns = data_one.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'one_' + str(columns[col])

    columns = data_two.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'two_' + str(columns[col])

    data = pandas.concat([data_one, data_two, data_three], axis=1, sort=False)

    data = data.sample(frac=1, random_state=random_state)

    labels = data[['class']]
    data.drop(['class'], axis=1, inplace=True)

    return data, labels


def read_in_csv_data_2(source_one, source_two):
    data_one = pandas.read_csv(source_one, sep=',')
    data_two = pandas.read_csv(source_two, sep=',')
    data_two.drop(['class'], axis=1, inplace=True)

    columns = data_one.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'one_' + str(columns[col])

    data = pandas.concat([data_one, data_two], axis=1, sort=False)

    data = data.sample(frac=1, random_state=random_state)

    labels = data[['class']]
    data.drop(['class'], axis=1, inplace=True)

    return data, labels


comb = 2


def main():
    data = None
    labels = None
    if comb == 2:
        data, labels = read_in_csv_data_2(get_file_base() + 'extracted_features/OVR/lda_stemmed_OVR.csv',
                                          get_file_base() + 'extracted_features/OVR/tfidf_stemmed_OVR.csv')
    if comb == 3:
        data, labels = read_in_csv_data_3(get_file_base() + 'extracted_features/tfidf_cos_unstemmed.csv',
                                          get_file_base() + 'extracted_features/bert_cos_unstemmed.csv',
                                          get_file_base() + 'extracted_features/lda_was_unstemmed.csv')
    if comb == 4:
        data, labels = read_in_csv_data_4(get_file_base() + 'extracted_features/tfidf_cos_unstemmed.csv',
                                          get_file_base() + 'extracted_features/bert_cos_unstemmed.csv',
                                          get_file_base() + 'extracted_features/lda_was_unstemmed.csv',
                                          get_file_base() + 'extracted_features/years.csv')

    print('ALL FEATURES')
    all_single_feature_classify_data(data, labels)


if __name__ == '__main__':
    main()
