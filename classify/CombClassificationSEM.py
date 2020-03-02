import pandas
from .ClassificationSEM import all_single_feature_classify_data
from general.baseFileExtractor import get_file_base

random_state = 100


def read_in_csv_data_5(one_stemmed, two_stemmed, one_met, two_met, three_met, four_met):
    source_one = get_file_base() + 'extracted_features/tfidf_' + one_met + '_' + one_stemmed + '.csv'
    source_two = get_file_base() + 'extracted_features/lda_' + two_met + '_' + two_stemmed + '.csv'
    source_three = get_file_base() + 'extracted_features/d2v_' + three_met + '_unstemmed.csv'
    source_four = get_file_base() + 'extracted_features/bert_' + four_met + '_unstemmed.csv'
    source_five = get_file_base() + 'extracted_features/years_dist_unstemmed.csv'

    data_one = pandas.read_csv(source_one, sep=',')
    data_two = pandas.read_csv(source_two, sep=',')
    data_three = pandas.read_csv(source_three, sep=',')
    data_four = pandas.read_csv(source_four, sep=',')
    data_five = pandas.read_csv(source_five, sep=',')

    data_two.drop(['class'], axis=1, inplace=True)
    data_three.drop(['class'], axis=1, inplace=True)
    data_four.drop(['class'], axis=1, inplace=True)
    data_five.drop(['class'], axis=1, inplace=True)

    columns = data_one.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'one_' + str(columns[col])

    columns = data_two.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'two_' + str(columns[col])

    columns = data_three.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'three_' + str(columns[col])

    columns = data_four.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'four_' + str(columns[col])

    data = pandas.concat([data_one, data_two, data_three, data_four, data_five], axis=1, sort=False)

    data = data.sample(frac=1, random_state=random_state)

    labels = data[['class']]
    data.drop(['class'], axis=1, inplace=True)

    return data, labels


def read_in_csv_data_4(one_4, one_met, one_stemmed, two_4, two_met, two_stemmed, three_4, three_met, three_stemmed,
                       four_4, four_met, four_stemmed):
    source_one = get_file_base() + 'extracted_features/' + one_4 + '_' + one_met + '_' + one_stemmed + '.csv'
    source_two = get_file_base() + 'extracted_features/' + two_4 + '_' + two_met + '_' + two_stemmed + '.csv'
    source_three = get_file_base() + 'extracted_features/' + three_4 + '_' + three_met + '_' + three_stemmed + '.csv'
    source_four = get_file_base() + 'extracted_features/' + four_4 + '_' + four_met + '_' + four_stemmed + '.csv'

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


def read_in_csv_data_3(one_3, one_met, one_stemmed, two_3, two_met, two_stemmed, three_3, three_met, three_stemmed):
    source_one = get_file_base() + 'extracted_features/' + one_3 + '_' + one_met + '_' + one_stemmed + '.csv'
    source_two = get_file_base() + 'extracted_features/' + two_3 + '_' + two_met + '_' + two_stemmed + '.csv'
    source_three = get_file_base() + 'extracted_features/' + three_3 + '_' + three_met + '_' + three_stemmed + '.csv'

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


def read_in_csv_data_2(one_2, one_met, one_stemmed, two_2, two_met, two_stemmed):
    source_one = get_file_base() + 'extracted_features/' + one_2 + '_' + one_met + '_' + one_stemmed + '.csv'
    source_two = get_file_base() + 'extracted_features/' + two_2 + '_' + two_met + '_' + two_stemmed + '.csv'

    data_one = pandas.read_csv(source_one, sep=',')
    data_two = pandas.read_csv(source_two, sep=',')
    data_two.drop(['class'], axis=1, inplace=True)

    columns = data_one.columns

    for col in range(1, len(columns)):
        columns.values[col] = 'one_' + str(columns[col])

    data = pandas.concat([data_one, data_two, ], axis=1, sort=False)

    data = data.sample(frac=1, random_state=random_state)

    labels = data[['class']]
    data.drop(['class'], axis=1, inplace=True)

    return data, labels


# [[distance measures per document vector embedding], boolean if stemmed version is available]
tfidf = [['cos', 'jac', 'ipd'], True]
d2v = [['cos', 'jac', 'ipd'], False]
bert = [['cos', 'jac', 'ipd'], False]
lda = [['emd', 'ipd'], True]
years = [['dist'], False]

########################################################################################################################
# configuration
text_one = 'lda'
text_two = 'd2v'
text_three = 'bert'
text_four = 'years'

comb = 5
########################################################################################################################


def main():
    one = ''
    if text_one == 'tfidf':
        one = tfidf
    if text_one == 'd2v':
        one = d2v
    if text_one == 'bert':
        one = bert
    if text_one == 'lda':
        one = lda
    if text_one == 'years':
        one = years

    two = ''
    if text_two == 'tfidf':
        two = tfidf
    if text_two == 'd2v':
        two = d2v
    if text_two == 'bert':
        two = bert
    if text_two == 'lda':
        two = lda
    if text_two == 'years':
        two = years

    three = ''
    if text_three == 'tfidf':
        three = tfidf
    if text_three == 'd2v':
        three = d2v
    if text_three == 'bert':
        three = bert
    if text_three == 'lda':
        three = lda
    if text_three == 'years':
        three = years

    four = ''
    if text_four == 'tfidf':
        four = tfidf
    if text_four == 'd2v':
        four = d2v
    if text_four == 'bert':
        four = bert
    if text_four == 'lda':
        four = lda
    if text_four == 'years':
        four = years

    if comb == 2:
        for mes_one in one[0]:
            for mes_two in two[0]:
                print('################################################################')

                if one[1] and two[1]:
                    print(text_one + '_' + mes_one + '_stemmed - ' + text_two + '_' + mes_two + '_stemmed:')

                    data, labels = read_in_csv_data_2(text_one, mes_one, 'stemmed', text_two, mes_two, 'stemmed')
                    all_single_feature_classify_data(data, labels, None)

                if one[1]:
                    print('................................................................')
                    print(text_one + '_' + mes_one + '_stemmed - ' + text_two + '_' + mes_two + '_unstemmed:')

                    data, labels = read_in_csv_data_2(text_one, mes_one, 'stemmed', text_two, mes_two, 'unstemmed')
                    all_single_feature_classify_data(data, labels, None)

                if two[1]:
                    print('................................................................')
                    print(text_one + '_' + mes_one + '_unstemmed - ' + text_two + '_' + mes_two + '_stemmed:')

                    data, labels = read_in_csv_data_2(text_one, mes_one, 'unstemmed', text_two, mes_two, 'stemmed')
                    all_single_feature_classify_data(data, labels, None)

                print('................................................................')
                print(text_one + '_' + mes_one + '_unstemmed - ' + text_two + '_' + mes_two + '_unstemmed:')

                data, labels = read_in_csv_data_2(text_one, mes_one, 'unstemmed', text_two, mes_two, 'unstemmed')
                all_single_feature_classify_data(data, labels, None)

    if comb == 3:
        for mes_one in one[0]:
            for mes_two in two[0]:
                for mes_three in three[0]:
                    print('################################################################')

                    if one[1] and two[1]:
                        if three[1]:
                            print(text_one + '_' + mes_one + '_stemmed - ' +
                                  text_two + '_' + mes_two + '_stemmed - ' +
                                  text_three + '_' + mes_three + '_stemmed:')

                            data, labels = read_in_csv_data_3(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                              'stemmed', text_three, mes_three, 'stemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        print(text_one + '_' + mes_one + '_stemmed - ' +
                              text_two + '_' + mes_two + '_stemmed - ' +
                              text_three + '_' + mes_three + '_unstemmed:')

                        data, labels = read_in_csv_data_3(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                          'stemmed', text_three, mes_three, 'unstemmed')
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

                    if one[1] and three[1]:
                        print(text_one + '_' + mes_one + '_stemmed - ' +
                              text_two + '_' + mes_two + '_unstemmed - ' +
                              text_three + '_' + mes_three + '_stemmed:')

                        data, labels = read_in_csv_data_3(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                          'unstemmed', text_three, mes_three, 'stemmed')
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

                    if two[1] and three[1]:
                        print(text_one + '_' + mes_one + '_unstemmed - ' +
                              text_two + '_' + mes_two + '_stemmed - ' +
                              text_three + '_' + mes_three + '_stemmed:')

                        data, labels = read_in_csv_data_3(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                          'stemmed', text_three, mes_three, 'stemmed')
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

                    if one[1]:
                        print(text_one + '_' + mes_one + '_stemmed - ' +
                              text_two + '_' + mes_two + '_unstemmed - ' +
                              text_three + '_' + mes_three + '_unstemmed:')

                        data, labels = read_in_csv_data_3(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                          'unstemmed', text_three, mes_three, 'unstemmed')
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

                    if two[1]:
                        print(text_one + '_' + mes_one + '_unstemmed - ' +
                              text_two + '_' + mes_two + '_stemmed - ' +
                              text_three + '_' + mes_three + '_unstemmed:')

                        data, labels = read_in_csv_data_3(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                          'stemmed', text_three, mes_three, 'unstemmed')
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

                    if three[1]:
                        print(text_one + '_' + mes_one + '_unstemmed - ' +
                              text_two + '_' + mes_two + '_unstemmed - ' +
                              text_three + '_' + mes_three + '_stemmed:')

                        data, labels = read_in_csv_data_3(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                          'unstemmed', text_three, mes_three, 'stemmed')
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

                    print(text_one + '_' + mes_one + '_unstemmed - ' +
                          text_two + '_' + mes_two + '_unstemmed - ' +
                          text_three + '_' + mes_three + '_unstemmed:')

                    data, labels = read_in_csv_data_3(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                      'unstemmed', text_three, mes_three, 'unstemmed')
                    all_single_feature_classify_data(data, labels, None)

                    print('................................................................')

    if comb == 4:
        for mes_one in one[0]:
            for mes_two in two[0]:
                for mes_three in three[0]:
                    for mes_four in four[0]:
                        print('################################################################')

                        if one[1] and two[1] and three[1]:
                            if four[1]:
                                print(text_one + '_' + mes_one + '_stemmed - ' +
                                      text_two + '_' + mes_two + '_stemmed - ' +
                                      text_three + '_' + mes_three + '_stemmed - ' +
                                      text_four + '_' + mes_four + '_stemmed:')

                                data, labels = read_in_csv_data_4(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                                  'stemmed', text_three, mes_three, 'stemmed',
                                                                  text_four, mes_four, 'stemmed')
                                all_single_feature_classify_data(data, labels, None)

                                print('................................................................')

                            print(text_one + '_' + mes_one + '_stemmed - ' +
                                  text_two + '_' + mes_two + '_stemmed - ' +
                                  text_three + '_' + mes_three + '_stemmed - ' +
                                  text_four + '_' + mes_four + '_unstemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                              'stemmed', text_three, mes_three, 'stemmed',
                                                              text_four, mes_four, 'unstemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if one[1] and two[1] and four[1]:
                            print(text_one + '_' + mes_one + '_stemmed - ' +
                                  text_two + '_' + mes_two + '_stemmed - ' +
                                  text_three + '_' + mes_three + '_unstemmed - ' +
                                  text_four + '_' + mes_four + '_stemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                              'stemmed', text_three, mes_three, 'unstemmed',
                                                              text_four, mes_four, 'stemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if one[1] and two[1]:
                            print(text_one + '_' + mes_one + '_stemmed - ' +
                                  text_two + '_' + mes_two + '_stemmed - ' +
                                  text_three + '_' + mes_three + '_unstemmed - ' +
                                  text_four + '_' + mes_four + '_unstemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                              'stemmed', text_three, mes_three, 'unstemmed',
                                                              text_four, mes_four, 'unstemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if one[1] and three[1] and four[1]:
                            print(text_one + '_' + mes_one + '_stemmed - ' +
                                  text_two + '_' + mes_two + '_unstemmed - ' +
                                  text_three + '_' + mes_three + '_stemmed - ' +
                                  text_four + '_' + mes_four + '_stemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                              'unstemmed', text_three, mes_three, 'stemmed',
                                                              text_four, mes_four, 'stemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if one[1] and three[1]:
                            print(text_one + '_' + mes_one + '_stemmed - ' +
                                  text_two + '_' + mes_two + '_unstemmed - ' +
                                  text_three + '_' + mes_three + '_stemmed - ' +
                                  text_four + '_' + mes_four + '_unstemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                              'unstemmed', text_three, mes_three, 'stemmed',
                                                              text_four, mes_four, 'unstemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if one[1] and four[1]:
                            print(text_one + '_' + mes_one + '_stemmed - ' +
                                  text_two + '_' + mes_two + '_unstemmed - ' +
                                  text_three + '_' + mes_three + '_unstemmed - ' +
                                  text_four + '_' + mes_four + '_stemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                              'unstemmed', text_three, mes_three, 'unstemmed',
                                                              text_four, mes_four, 'stemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if one[1]:
                            print(text_one + '_' + mes_one + '_stemmed - ' +
                                  text_two + '_' + mes_two + '_unstemmed - ' +
                                  text_three + '_' + mes_three + '_unstemmed - ' +
                                  text_four + '_' + mes_four + '_unstemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'stemmed', text_two, mes_two,
                                                              'unstemmed', text_three, mes_three, 'unstemmed',
                                                              text_four, mes_four, 'unstemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if two[1] and three[1] and four[1]:
                            print(text_one + '_' + mes_one + '_unstemmed - ' +
                                  text_two + '_' + mes_two + '_stemmed - ' +
                                  text_three + '_' + mes_three + '_stemmed - ' +
                                  text_four + '_' + mes_four + '_stemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                              'stemmed', text_three, mes_three, 'stemmed',
                                                              text_four, mes_four, 'stemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if two[1] and three[1]:
                            print(text_one + '_' + mes_one + '_unstemmed - ' +
                                  text_two + '_' + mes_two + '_stemmed - ' +
                                  text_three + '_' + mes_three + '_stemmed - ' +
                                  text_four + '_' + mes_four + '_unstemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                              'stemmed', text_three, mes_three, 'stemmed',
                                                              text_four, mes_four, 'unstemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if two[1] and four[1]:
                            print(text_one + '_' + mes_one + '_unstemmed - ' +
                                  text_two + '_' + mes_two + '_stemmed - ' +
                                  text_three + '_' + mes_three + '_unstemmed - ' +
                                  text_four + '_' + mes_four + '_stemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                              'stemmed', text_three, mes_three, 'unstemmed',
                                                              text_four, mes_four, 'stemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if two[1]:
                            print(text_one + '_' + mes_one + '_unstemmed - ' +
                                  text_two + '_' + mes_two + '_stemmed - ' +
                                  text_three + '_' + mes_three + '_unstemmed - ' +
                                  text_four + '_' + mes_four + '_unstemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                              'stemmed', text_three, mes_three, 'unstemmed',
                                                              text_four, mes_four, 'unstemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if three[1] and four[1]:
                            print(text_one + '_' + mes_one + '_unstemmed - ' +
                                  text_two + '_' + mes_two + '_unstemmed - ' +
                                  text_three + '_' + mes_three + '_stemmed - ' +
                                  text_four + '_' + mes_four + '_stemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                              'unstemmed', text_three, mes_three, 'stemmed',
                                                              text_four, mes_four, 'stemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if three[1]:
                            print(text_one + '_' + mes_one + '_unstemmed - ' +
                                  text_two + '_' + mes_two + '_unstemmed - ' +
                                  text_three + '_' + mes_three + '_stemmed - ' +
                                  text_four + '_' + mes_four + '_unstemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                              'unstemmed', text_three, mes_three, 'stemmed',
                                                              text_four, mes_four, 'unstemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        if four[1]:
                            print(text_one + '_' + mes_one + '_unstemmed - ' +
                                  text_two + '_' + mes_two + '_unstemmed - ' +
                                  text_three + '_' + mes_three + '_unstemmed - ' +
                                  text_four + '_' + mes_four + '_stemmed:')

                            data, labels = read_in_csv_data_4(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                              'unstemmed', text_three, mes_three, 'unstemmed',
                                                              text_four, mes_four, 'stemmed')
                            all_single_feature_classify_data(data, labels, None)

                            print('................................................................')

                        print(text_one + '_' + mes_one + '_unstemmed - ' +
                              text_two + '_' + mes_two + '_unstemmed - ' +
                              text_three + '_' + mes_three + '_unstemmed - ' +
                              text_four + '_' + mes_four + '_unstemmed:')

                        data, labels = read_in_csv_data_4(text_one, mes_one, 'unstemmed', text_two, mes_two,
                                                          'unstemmed', text_three, mes_three, 'unstemmed',
                                                          text_four, mes_four, 'unstemmed')
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

    if comb == 5:
        one = tfidf
        two = lda
        three = d2v
        four = bert

        for mes_one in one[0]:
            for mes_two in two[0]:
                for mes_three in three[0]:
                    for mes_four in four[0]:
                        # 0 - 0 - 0 - 0 - 0
                        print('tfidf_' + mes_one + '_unstemmed - ' +
                              'lda_' + mes_two + '_unstemmed - ' +
                              'd2v_' + mes_three + '_unstemmed - ' +
                              'bert_' + mes_three + '_unstemmed - ' +
                              'years_dist_unstemmed:')

                        data, labels = read_in_csv_data_5('unstemmed', 'unstemmed', mes_one, mes_two, mes_three,
                                                          mes_four)
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

                        # 1 - 0 - 0 - 0 - 0
                        print('tfidf_' + mes_one + '_stemmed - ' +
                              'lda_' + mes_two + '_unstemmed - ' +
                              'd2v_' + mes_three + '_unstemmed - ' +
                              'bert_' + mes_three + '_unstemmed - ' +
                              'years_dist_unstemmed:')

                        data, labels = read_in_csv_data_5('stemmed', 'unstemmed', mes_one, mes_two, mes_three,
                                                          mes_four)
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')

                        # 1 - 1 - 0 - 0 - 0
                        print('tfidf_' + mes_one + '_stemmed - ' +
                              'lda_' + mes_two + '_stemmed - ' +
                              'd2v_' + mes_three + '_unstemmed - ' +
                              'bert_' + mes_three + '_unstemmed - ' +
                              'years_dist_unstemmed:')

                        data, labels = read_in_csv_data_5('stemmed', 'stemmed', mes_one, mes_two, mes_three,
                                                          mes_four)
                        all_single_feature_classify_data(data, labels, None)

                        print('................................................................')


if __name__ == '__main__':
    main()
