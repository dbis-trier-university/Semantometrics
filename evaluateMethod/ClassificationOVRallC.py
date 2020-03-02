from classify.ClassificationSEMallC import read_in_csv_data, all_single_feature_classify_data
from general.baseFileExtractor import get_file_base, get_stem


def restrict_data(data, part):
    third = int(len(data.columns) / 3)
    to_drop = []

    # delete p and citations, keep references
    if part == 1:
        for x in range(third, 3*third):
            to_drop.append(str(x))

        data.drop(to_drop, axis=1, inplace=True)

    # delete p and references, keep citations
    if part == 2:
        for x in range(0, 2*third):
            to_drop.append(str(x))

        data.drop(to_drop, axis=1, inplace=True)

    # delete citations, keep references and p
    if part == 3:
        for x in range(2*third, 3*third):
            to_drop.append(str(x))

        data.drop(to_drop, axis=1, inplace=True)

    # only p
    if part == 4:
        # delete references
        for x in range(0, third):
            to_drop.append(str(x))

        # delete citations
        for x in range(2*third, 3*third):
            to_drop.append(str(x))

        data.drop(to_drop, axis=1, inplace=True)

    return data, third


def drop_rs(data, third, use_stemming):
    if third == -1:
        third = int(len(data.columns) / 3)

    if use_stemming:
        data.drop([str(third + 380), str(third + 1105)], axis=1, inplace=True)
    else:
        data.drop([str(third + 460), str(third + 1576)], axis=1, inplace=True)

    return data


def main():
    # 0 = complete citation network, 1 = only references, 2 = only citations, 3 = references and p, 4 = only p
    only_part = 4
    use_stemming = get_stem()
    wo_RS = False

    # read in normal distance features for all publications
    data, labels = read_in_csv_data(get_file_base() + 'extracted_features/OVR/years_unstemmed_OVR.csv')
    print('ALL FEATURES')

    third = -1
    if only_part == 1:
        data, third = restrict_data(data, 1)
    if only_part == 2:
        data, third = restrict_data(data, 2)
    if only_part == 3:
        data, third = restrict_data(data, 3)

        if wo_RS:
            data = drop_rs(data, third, use_stemming)
    if only_part == 4:
        data, third = restrict_data(data, 4)

    if only_part == 0 and wo_RS:
        data = drop_rs(data, third, use_stemming)

    all_single_feature_classify_data(data, labels)


if __name__ == '__main__':
    main()
