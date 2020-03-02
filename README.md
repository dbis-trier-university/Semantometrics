# SEMANTOMETRICS

This is a Python project for ternary semantometrics: classification of publications as *seminal*, *survey* or *uninfluential*.

---
## Usage

Base files from the area of computer science (SUSdblp) available under: https://zenodo.org/record/3693939#.Xl0cF0oxlEa

- open **specification.txt** in folder **general**
- specify path to base files
- specify what you want to compute
- run **run.py** (in package general)

---
## Specification what to compute 

#### 0 = create all vector representations
#### 1 = create a single vector representation
specify which vectors should be constructed (via **vectors**):

- TFIDF-s: stemmed version of TF-IDF vectors, set *stem*=True
- TFIDF-u: unstemmed version of TF-IDF vectors, set *stem*=False
- D2V: unstemmed Doc2Vec vectors (additionally, you need to specify, where your Wikipedia file for the construction of the Doc2Vec model lies via *path_d2v_base*)
- BERT: unstemmed BERT vectors
- LDA-s: stemmed version of LDA vectors (additionally, you need to specify, where your unstemmed base txt-file for the construction of the LDA model lies via *path_lda_base*), set *stem*=True
- LDA-u: unstemmed version of LDA vectors (additionally, you need to specify, where your unstemmed base txt-file for the construction of the LDA model lies via *path_lda_base*), set *stem*=False

#### 2 = construct all features for semantometrics
#### 3 = construct one feature for semantometrics
specify which vectors (via **vectors**) 

- TFIDF-s: stemmed version of TF-IDF vectors, set *stem*=True
- TFIDF-u: unstemmed version of TF-IDF vectors, set *stem*=False
- D2V: unstemmed Doc2Vec vectors
- BERT: unstemmed BERT vectors
- LDA-s: stemmed version of LDA vectors, set *stem*=True
- LDA-u: unstemmed version of LDA vectors, set *stem*=False
- years: usage of years of publications 
	
and which distance (via **distance_measure**) should be used:

- cos: cosine distance
- jac: Jaccard distance
- emd: earth mover's distance
- ipd: inner product distance

#### 4 = all classifier classifications on semantometrics
specify from which vectors (via **vectors**):

- TFIDF-s: stemmed version of TF-IDF vectors, set *stem*=True
- TFIDF-u: unstemmed version of TF-IDF vectors, set *stem*=False
- D2V: unstemmed Doc2Vec vectors
- BERT: unstemmed BERT vectors
- LDA-s: stemmed version of LDA vectors, set *stem*=True
- LDA-u: unstemmed version of LDA vectors, set *stem*=False
- years: usage of years of publications 
		
with which distance (via **dist_measure**):
- cos: cosine distance
- jac: Jaccard distance
- emd: earth mover's distance
- ipd: inner product distance

#### 5 = single classifier classification on semantometrics
specify from which vectors (via **vectors**):

- TFIDF-s: stemmed version of TF-IDF vectors, set *stem*=True
- TFIDF-u: unstemmed version of TF-IDF vectors, set *stem*=False
- D2V: unstemmed Doc2Vec vectors
- BERT: unstemmed BERT vectors
- LDA-s: stemmed version of LDA vectors, set *stem*=True
- LDA-u: unstemmed version of LDA vectors, set *stem*=False
- years: usage of years of publications 
	
with which distance (via **dist_measure**):

- cos: cosine distance
- jac: Jaccard distance
- emd: earth mover's distance
- ipd: inner product distance<a/>

and which classifier (via **classifier**):

- LR: logistic regression
- RF: random forest
- NB: naive bayes
- SVM: support vector machines
- GB: gradient boosting
- KNN: k nearest neighbors
- SGD: stochastical gradient descent

#### 6 = all classifier classifications on semantometrics with information available at publication time, now specify from which vectors

specify from which vectors (via **vectors**):

- TFIDF-s: stemmed version of TF-IDF vectors, set *stem*=True
- TFIDF-u: unstemmed version of TF-IDF vectors, set *stem*=False
- D2V: unstemmed Doc2Vec vectors
- BERT: unstemmed BERT vectors
- LDA-s: stemmed version of LDA vectors, set *stem*=True
- LDA-u: unstemmed version of LDA vectors, set *stem*=False
- years: usage of years of publications 

with which distance (via **dist_measure**):

- cos: cosine distance
- jac: Jaccard distance
- emd: earth mover's distance
- ipd: inner product distance

#### 7 = single classifier classification on semantometrics with information available at publication time

specify from which vectors (via **vectors**):

- TFIDF-s: stemmed version of TF-IDF vectors, set *stem*=True
- TFIDF-u: unstemmed version of TF-IDF vectors, set *stem*=False
- D2V: unstemmed Doc2Vec vectors
- BERT: unstemmed BERT vectors
- LDA-s: stemmed version of LDA vectors, set *stem*=True
- LDA-u: unstemmed version of LDA vectors, set *stem*=False
- years: usage of years of publications 

with which distance (via **dist_measure**):

- cos: cosine distance
- jac: Jaccard distance
- emd: earth mover's distance
- ipd: inner product distance


and which classifier (via **classifier**):

- LR: logistic regression
- RF: random forest
- NB: naive bayes
- SVM: support vector machines
- GB: gradient boosting
- KNN: k nearest neighbors
- SGD: stochastic gradient descent	


---
## Package Descriptions


---
### analyseDataset
Provides options of analysing the specified dataset in terms of distributions of years, numbers of references and citations as well as boxplots of semantometic features or distributions.

---
### classify

Provides options of classification based on semantometric features:

- ClassificationSEM.py: Classification based on single or multiple (all or those which are available at publication time) semantometric features with a single predefined classifier. In general, the possibility of classification on OVR is also provided.
- ClassificationSEMallC.py:C lassification based on single or multiple (all, those which are significant in Herrmannova's experiments or those which are available at publication time) semantometric features with all seven classifiers. In general, the possibility of accessment of importance of features is also provided.
- CombClassification.py: Classification on all features derived from a predefined combination of 2, 3 or 4 document vector representations. OVR vectors can also be combined here.
- CombClassificationSEM.py: Classification on all sementomeric features derived from a predefined combination of 2, 3, 4 or 5 document vector representations.
- GetImportantFeaturesInClassification.py: Calculation of the 10 best features for classification based on all features from a predefined document vector representation, distance measure as well as classifier.

---
### computeFeatures

Provides options of semantometic feature calculations based on combinations of document vector representation and distance measure

---
### evaluateDataset
Provides options of evaluation the dataset such as topical composition of base files, abstract length bias, classification performance in different years and robustness to omission of references/citations.

---
### evaluateMethod
Provides options of classification on non-semantometric-features such as the number of references and citations or one vector representations.

---
### general
Provides script which extracts specifications of users from specification.txt which is also located in the same folder.

---
### generateData
Used for generation of TF-IDF, Doc2Vec, BERT and LDA vector representations of publications. Generation of Doc2Vec and LDA model.

#### TF-IDF Vectors
- class: TFIDFEmbedding.py
- does: generates TF-IDF vectors for (un)stemmed base files
- takes: three json-files of (un)stemmed base data
- returns: TF-IDF vectors of (un)stemmed base files in one sav-file, words corresponding to dimensions


#### Doc2Vec Vectors:

- classes: BuildD2VModel.py, D2VEmbedding.py
- does: trains Doc2Vec model from Wikipedia base and generates vectors for unstemmed base files with it
- takes: Wikipedia corups (via path_d2v_base.txt), three json-files of unstemmed base data
- returns: D2V model, Doc2Vec vectors of unstemmed base files in one pickle-file

#### BERT Vectors:

- class: BERTEmbedding.py
- does: generates BERT vectors, uses pretrained uncased BERT model
- takes: three json-files of unstemmed base data
- returns: a json-file for each of the three classes

### LDA Vectors:

- classes: BuildLDAModel.py, LDAEmbedding.py
- does: trains LDA model with 100 topics with and generates vectors for unstemmed base files with it
- takes: (un)stemmed LDA training corpus in txt-file, each line represents a new document (via path_lda_base.txt)
- returns: LDA model, dictionary, json-file for each of the three classes, json-file for each of the three classes as one-document-representation
