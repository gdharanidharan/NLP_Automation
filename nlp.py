import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import scikitplot as skplt

# text data - NLP problem
# apply stemming and lemmatization
# embedding - fidf
# train test split
# model building
# model evaluation

class NLP:

    def __init__(self, data):
        self.data = data

# stemming
    def stemming(self, column_name):

        '''
        cleaning data using re, porter stemmer and stopwords
        '''
        
        try:
            corpus = []
            stemming = PorterStemmer()
            for i in range(len(self.data)):
                tweet = re.sub('[^a-zA-Z]', ' ', self.data[column_name][i])
                tweet = re.sub('http', '', tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [stemming.stem(word) for word in tweet if word not in set(stopwords.words('english'))]
                tweet = ' '.join(tweet)
                corpus.append(tweet)

        except Exception as e:
            print("Stemming Error:", e)

        else:
            print('Data cleaning completed')
            return corpus


 # lemmatization 
    def lemmatization(self, column_name):

        '''
        cleaning data using re, lemmatization and stopwords
        '''
        
        try:
            corpus = []
            lemmatizer = WordNetLemmatizer()
            for i in range(len(self.data)):
                tweet = re.sub('[^a-zA-Z]', ' ', self.data[column_name][i])
                tweet = re.sub('http', '', tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [lemmatizer.lemmatize(word) for word in tweet if word not in set(stopwords.words('english'))]
                tweet = ' '.join(tweet)
                corpus.append(tweet)

        except Exception as e:
            print("Lemmatization Error:", e)

        else:
            print('Data cleaning completed')
            return corpus


# count vectorizer
    def count_vectorizer(self, corpus, max_features=3000, ngram_range=(1,2)):

        '''
        count vectorizer
        '''

        try:
            cv = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            X = cv.fit_transform(corpus).toarray()

        except Exception as e:
            print('Count Vectorizer error ',e)

        else:
            print('vectorizing using BOW completed')
            return X


# Tfidf vectorizer
    def tfidf_vectorizer(self, corpus, max_features=3000, ngram_range=(1,2)):

        '''
        Tfidf vectorizer
        '''

        try:
            tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            X = tfidf.fit_transform(corpus).toarray()

        except Exception as e:
            print('Tfidf Vectorizer error ',e)

        else:
            print('vectorizing using tfidf completed')
            return X


# encoding target feature
    def y_encoding(self, target_label):

        '''
        encoding target label using pandas get dummies
        '''
        try:
            y = pd.get_dummies(self.data[target_label], drop_first=True)
            
        except Exception as e:
            print('Target Encoding Error', e)

        else:
            print('Target label encoding completed')
            return y


# train test split
    def split_data(self, X, y, test_size=0.25, random_state=0):

        '''
        spliting into training and testing data
        '''

        try:
            X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, test_size=test_size, random_state=random_state)
        
        except Exception as e:
            print('Train Test Split Error', e)

        else:
            print('Splitting Data Completed')
            return X_train, X_test, y_train, y_test


# model training
    def navie_bayes_model(self, X_train, X_test, y_train, y_test):

        try:
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
        except Exception as e:
            print('Model Training and Prediction Error', e)

        else:
            return y_pred

# model evaluation
    def confusion_matrix_accuracy(self, y_test, y_pred):

        try:
            skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(8,7))
            plt.savefig('confusion_matrix.jpg')
            img_cm = Image.open('confusion_matrix.jpg')
            accuracy = accuracy_score(y_test, y_pred)

        except Exception as e:
            print('Model Evaluation Error', e)
        
        else:
            return accuracy, img_cm

# word cloud
    def word_cloud(self, corpus):
        
        try:
            wordcloud = wordcloud(
                background_color='white',
                width=750,
                height=500
            ).generate(" ".join(corpus))

            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig('wordcloud.jpg')
            img_wc = Image.open('wordcloud.jpg')

        except Exception as e:
            print('Word Cloud Error', e)

        else:
            return img_wc  
