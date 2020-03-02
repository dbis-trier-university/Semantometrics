from .ClassificationSEM import read_in_csv_data
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from general.baseFileExtractor import get_file_base

folds = 10
number_of_neighbors = 5
random_state = 100

# LR
clf_lr = LogisticRegression(solver='lbfgs', random_state=random_state, max_iter=1000, penalty='l2',
                            fit_intercept=True, multi_class='ovr')

# RF
clf_rf = RandomForestClassifier(n_estimators=100, random_state=random_state, criterion='entropy', max_features='auto')

# NB
clf_nb = GaussianNB()

# SVM
clf_svm = SVC(gamma='scale', random_state=random_state)

# GB
clf_gb = GradientBoostingClassifier(random_state=random_state, loss='deviance', learning_rate=0.025,
                                    n_estimators=200)

# KNN
clf_knn = KNeighborsClassifier(n_neighbors=number_of_neighbors)

# SGD
clf_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=random_state, tol=None)


def classify(X, y, clf):
    y = y.astype('int')
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

    # get correlations of each features in dataset
    corrmat = X.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(X[top_corr_features].corr(), annot=False, cmap="coolwarm")
    figure = g.get_figure()
    figure.savefig(get_file_base() + 'plots\importance_heatmap.png')

    bestfeatures = SelectKBest(score_func=f_classif, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(10, 'Score'))  # print 10 best features


def main():
    # read in normal distance features for all publications
    data, labels = read_in_csv_data(get_file_base() + 'extracted_features/years_dist_unstemmed.csv')
    classify(data, labels, clf_gb)


if __name__ == '__main__':
    main()
