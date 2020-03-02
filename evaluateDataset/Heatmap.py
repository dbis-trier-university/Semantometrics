import seaborn as sns
from classify.ClassificationSEM import read_in_csv_data
from general.baseFileExtractor import get_file_base


data, labels = read_in_csv_data(get_file_base() + 'extracted_features/tfidf_cos_unstemmed.csv')

ax = sns.heatmap(data.corr())
figure = ax.get_figure()
figure.savefig(get_file_base() + 'plots/heatmap.png')