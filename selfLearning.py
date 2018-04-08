#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 23:27:02 2018

@author: pawan
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy 
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
def train_model():
	for i in range(0, 1000):
		text_data = dataset['Review'][i] 
		preprocessing(text_data)

	# Creating the Bag of Words model
	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer(max_features = 1500)
	X = cv.fit_transform(corpus).toarray()
	y = dataset.iloc[:, 1].values
	# Splitting the dataset into the Training set and Test set
	from sklearn.cross_validation import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size = 0.20,
		random_state = 0
		)
	print X_test
	print X_test.shape
	# print 
	# Fitting Naitve Bayes to the Training set
	from sklearn.naive_bayes import GaussianNB
	classifier = GaussianNB()
	classifier.fit(X_train, y_train)
	park_model(classifier,corpus)

def preprocessing(text):
	review = re.sub('[^a-zA-Z]', ' ',text)
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
	review = ' '.join(review)
	corpus.append(review)

def park_model(classifier,corpus):
	import pickle 
	review_predictor = {"classifier":classifier,"corpus":corpus}
	pickle_out = open("classifier.pickle","wb")
	pickle.dump(review_predictor, pickle_out)
	pickle_out.close()   

train_model()
# park_model(classifier)
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)