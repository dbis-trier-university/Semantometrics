from classify.ClassificationSEM import read_in_csv_data, all_single_feature_classify_data
from evaluateMethod.ClassificationOVRallC import restrict_data, drop_rs
from general.baseFileExtractor import get_file_base, get_stem


def main():
    # 0 = complete citation network, 1 = only references, 2 = only citations, 3 = references and p, 4 = only p
    only_part = 2
    use_stemming = get_stem()
    wo_RS = False
    classifier = 'GB'

    # read in normal distance features for all publications
    data, labels = read_in_csv_data(get_file_base() + 'extracted_features/OVR/bert_unstemmed_OVR.csv')
    print('ALL FEATURES')

    third = -1
    if only_part == 1:
        data, third = restrict_data(data, 1)
    if only_part == 2:
        data, third = restrict_data(data, 2)
    if only_part == 3:
        data, third = restrict_data(data, 3)
    if only_part == 4:
        data, third = restrict_data(data, 4)

        if wo_RS:
            data = drop_rs(data, third, use_stemming)

    if only_part == 0 and wo_RS:
        data = drop_rs(data, third, use_stemming)

    all_single_feature_classify_data(data, labels, classifier)


if __name__ == '__main__':
    main()
