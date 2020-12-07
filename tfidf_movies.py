"""
TF-IDF and Logistic Regression for Sentiment Analysis

Sources:
https://realpython.com/logistic-regression-python/#logistic-regression-in-python
https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
http://martinhjelm.github.io/2017/11/12/Pandas-Replacing-Strings-In-A-Column/
https://www.kaggle.com/codeserra09/twitter-us-airline-sentiment-lg-mnb-dt-rf-knn
https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
documentations: pandas, nltk, sklearn

@autor: Angelina Sonderecker
"""

import os
import pandas as pd
import nltk  # preprocessing
from nltk import PorterStemmer, WordNetLemmatizer, re
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer  # tokenization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report  # evaluation

# Change working directory to data
os.chdir('C:/Users/Angelina/Documents/#Master/3. Semester/Advanced Data Mining & Machine Learning/0-Assignment/data')

# Preprocess data
df = pd.read_csv('IMDB Dataset_truncated.csv')  # use 'IMDB Dataset_truncated.csv' for debugging
print(df.shape, df.head())

nltk.download('stopwords')
stopwords_english = stopwords.words('english')  # words without meaning
stemmer = PorterStemmer()  # to get word stem
nltk.download('wordnet')
lemma = WordNetLemmatizer()  # get meaningful word stem


def preprocess_tweet(text):
    new_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', text)  # remove URL
    new_tweet = re.sub(r'<[^>]+>', '', new_tweet)  # remove html (line breaks etc.)
    new_tweet = re.sub(re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"), '', new_tweet)  # remove email
    new_tweet = re.sub(r'#', '', new_tweet)  # remove hash sign from hashtags
    new_tweet = re.sub("[^a-zA-Z]", " ", new_tweet)  # remove remaining special characters
    words = new_tweet.lower().split()  # do lowercase, split into words
    words = [word for word in words if not word in stopwords_english]  # remove stop words
    words = [stemmer.stem(word) for word in words]  # stemming
    words = [lemma.lemmatize(word) for word in words]  # lemmatization
    # join words list back to one tweet
    return " ".join(words)


df['review'] = df['review'].apply(lambda x: preprocess_tweet(x))  # preprocess reviews
print(df.head())

# Split data in train and test data (default test_size=0.25, set random_state for reproducibility)
df_train, df_test = train_test_split(df)

# Tokenization: TF-IDF
vectorizer = TfidfVectorizer()
train = vectorizer.fit_transform(df_train['review'])  # use fit to scale training data, use this info for test data
test = vectorizer.transform(df_test['review'])

train_tfidf = pd.DataFrame(train.toarray(), columns=vectorizer.get_feature_names())  # only to view tfidf
print(vectorizer.get_feature_names())

# Classification: Logistic Regression
lr = LogisticRegression()
lr.fit(train, df_train['sentiment'])  # train model

# Classification: SVM
svm_clf = svm.SVC()
svm_clf.fit(train, df_train['sentiment'])

# Model evaluation: LR
lr_sentiment_pred = lr.predict(test)
lr_report = classification_report(df_test['sentiment'], lr_sentiment_pred)
print(lr_report)  # support: nr. occurences of labels in test

# Model evaluation: SVM
svm_sentiment_pred = svm_clf.predict(test)
svm_report = classification_report(df_test['sentiment'], svm_sentiment_pred)
print(svm_report)