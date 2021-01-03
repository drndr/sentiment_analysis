"""
TF-IDF with classifiers LR and SVM for Sentiment Analysis

Sources:
https://realpython.com/logistic-regression-python/#logistic-regression-in-python
https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
http://martinhjelm.github.io/2017/11/12/Pandas-Replacing-Strings-In-A-Column/
https://www.kaggle.com/codeserra09/twitter-us-airline-sentiment-lg-mnb-dt-rf-knn
https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
https://towardsdatascience.com/twitter-sentiment-analysis-using-fasttext-9ccd04465597
https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search/
https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
documentations: pandas, nltk, sklearn

@author: Angelina Sonderecker
"""
import sys
import os
import pandas as pd
import numpy as np
import nltk  # preprocessing
from nltk import PorterStemmer, WordNetLemmatizer, re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer  # tokenization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report  # evaluation
from sklearn.model_selection import GridSearchCV  # hyperparameter optimization

# Change working directory to data
os.chdir('C:/Users/Angelina/Documents/#Master/3. Semester/Advanced Data Mining & Machine Learning/0-Assignment/data')

# Select data
data_movies = False
if 'movies' in sys.argv:
    data_movies = True

if data_movies:
    df = pd.read_csv('IMDB Dataset.csv')  # use 'IMDB Dataset_truncated.csv' for debugging
    text = "review"
    sentiment = "sentiment"
else:
    df = pd.read_csv('Tweets.csv')  # use 'Tweets_truncated.csv' for debugging
    text = "text"
    sentiment = "airline_sentiment"

print(df.shape, df.head())

if not data_movies:  # only for airline dataset
    # view neutral sentiment
    neutral = df[df.airline_sentiment == "neutral"]["text"]
    print(neutral.head())
    # delete neutral sentiment
    df = df[df.airline_sentiment != "neutral"]  # exclude neutral entries
    print(df.shape, df.head())

# Preprocess data
nltk.download('stopwords')
stopwords_english = stopwords.words('english')  # words without meaning
stemmer = PorterStemmer()  # to get word stem
nltk.download('wordnet')
lemma = WordNetLemmatizer()  # get meaningful word stem


def preprocess_text(old_text):  # TODO emoji, emoticons and pycontractions (maybe lemmatization already does this)
    """preprocess given text and return str"""
    new_text = re.sub(r'https?:\/\/.*[\r\n]*', '', old_text)  # remove URL
    new_text = re.sub(r'<[^>]+>', '', new_text)  # remove html (line breaks etc.)
    new_text = re.sub(re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"), '',
                      new_text)  # remove email
    new_text = re.sub(r'#', '', new_text)  # remove hash sign from hashtags, hashtag itself remains
    # new_text = re.sub('@[^\s]+', '', new_text)  # deletes mentions with @ TODO I think it has no influence
    new_text = re.sub("[^a-zA-Z]", " ", new_text)  # remove remaining special characters
    words = new_text.lower().split()  # do lowercase, split into words
    words = [word for word in words if not word in stopwords_english]  # remove stop words
    words = [stemmer.stem(word) for word in words]  # stemming
    words = [lemma.lemmatize(word) for word in words]  # lemmatization
    # join words list back to one tweet
    return " ".join(words)


df['text_preprocessed'] = df[text].apply(lambda x: preprocess_text(x))
print(df.head())

# Split data in train and test data (default test_size=0.25, set random_state for reproducibility)
df_train, df_test = train_test_split(df, test_size=0.2)

# Tokenization using TF-IDF
vectorizer = TfidfVectorizer()
train = vectorizer.fit_transform(df_train['text_preprocessed'])
test = vectorizer.transform(df_test['text_preprocessed'])
train_label = df_train[sentiment]
test_label = df_test[sentiment]

# Classification and evaluation: Logistic Regression, default: C=1.0, solver='lbfgs', penalty='l2', max_iter=100
lr_clf = LogisticRegression(verbose=True)
lr_clf.fit(train, train_label)

lr_sentiment_pred = lr_clf.predict(test)
print("no optimization (lr):\n", classification_report(test_label, lr_sentiment_pred, digits=4))

# hyperparameter search space
# 'newton-cg', 'lbfgs', 'sag' only support 'l2' penalty
# 'liblinear' does not support 'none' penalty
# 'elasticnet' penalty works only for 'saga'
# solver = ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']  # lbfgs failed to converge, needs more iterations
# penalty = ['l1', 'l2', 'none']  # regularization penalty (l1=lasso, l2=ridge regression)
# max_iter = [1, 10, 50, 100, 1000]
# C = np.logspace(0, 4, 10)  # regularization strength (smaller means stronger regulaization), use logscale (10 is base)
C = [7.7426]
hyperparameters_lr = {'C': C}

# gird search, with crossvalidation=5
grid_search = GridSearchCV(estimator=lr_clf, param_grid=hyperparameters_lr, cv=5)
grid = grid_search.fit(train, train_label)

# Summarize results, best hyperparameters
print("Best (lr): %f using parameters %s,\n view model: %s" % (
    grid.best_score_, grid.best_params_, grid.best_estimator_))
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Predict using best model
lr_best_pred = grid.predict(test)
print("Best model (lr):\n", classification_report(test_label, lr_best_pred, digits=4))

###############################################################################

# Classification and evaluation: SVM
svm_clf = svm.SVC(verbose=True)
svm_clf.fit(train, train_label)

svm_sentiment_pred = svm_clf.predict(test)
print("no optimization (svm):\n", classification_report(test_label, svm_sentiment_pred, digits=4))

# hyperparameter search space
# C = np.logspace(0, 4, 10)  # default regularization C = 1.0 (penalty is squared l2)
# max_iter = [1, 50, 100, 1000, -1]  # default = -1 (no limit)
#gamma = ['scale', 'auto'] # default kernel coefficient = 'scale'
#kernel = ['linear', 'poly', 'rbf']  # default = 'rbf'
kernel = ['linear']

hyperparameters_svm = {'kernel': kernel}

# gird search, with crossvalidation=5
grid_search = GridSearchCV(estimator=svm_clf, param_grid=hyperparameters_svm, cv=5)
grid = grid_search.fit(train, train_label)

# Summarize results, best hyperparameters
print("Best (svm): %f using parameters %s,\n view model: %s" % (
    grid.best_score_, grid.best_params_, grid.best_estimator_))
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Predict using best model
svm_best_pred = grid.predict(test)
print("Best model (svm):\n", classification_report(test_label, svm_best_pred, digits=4))
