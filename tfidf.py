"""
TF-IDF with classifiers LR and SVM for Sentiment Analysis

Sources:
https://realpython.com/logistic-regression-python/#logistic-regression-in-python
https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
http://martinhjelm.github.io/2017/11/12/Pandas-Replacing-Strings-In-A-Column/
https://www.kaggle.com/codeserra09/twitter-us-airline-sentiment-lg-mnb-dt-rf-knn
https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
https://towardsdatascience.com/twitter-sentiment-analysis-using-fasttext-9ccd04465597
documentations: pandas, nltk, sklearn

@author: Angelina Sonderecker
"""
import sys
import os
import pandas as pd
import nltk  # preprocessing
from nltk import PorterStemmer, WordNetLemmatizer, re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer  # tokenization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report  # evaluation


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
    #new_text = re.sub('@[^\s]+', '', new_text)  # deletes mentions with @ TODO I think it has no influence
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


# TODO adjust parameters
# Classification and evaluation: Logistic Regression
lr_clf = LogisticRegression()
lr_clf.fit(train, df_train[sentiment])

lr_sentiment_pred = lr_clf.predict(test)
lr_report = classification_report(df_test[sentiment], lr_sentiment_pred)
print(lr_report)


# TODO adjust parameters, why does it take so long for movies to compute
# Classification and evaluation: SVM
svm_clf = svm.SVC()
svm_clf.fit(train, df_train[sentiment])

svm_sentiment_pred = svm_clf.predict(test)
svm_report = classification_report(df_test[sentiment], svm_sentiment_pred)
print(svm_report)