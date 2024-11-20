#naive bayes
#knn
#svm
#decision tree
#random forest
#logistic regression
#LDA
#QDA
#clubpenguin

#open and clean data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read in data (NEED TO TALK ABT THIS)
twitter = pd.read_csv('FakeNewsNet.csv') #ONLY INCLUDES TITLES OF ARTICLES
#print(twitter.head(10))
#twitter = twitter.drop(['news_url', 'source_domain', 'tweet_num'], axis=1)
print(twitter.head(10)) 

#read in albanian data
albanian = pd.read_csv('alb-fake-news-corpus.csv')
print(albanian.head(10))
print(albanian.columns)

fakeSoccer = pd.read_csv('fake-soccer.csv')
print(fakeSoccer.head(10))
print(fakeSoccer.columns)
realSoccer = pd.read_csv('real-soccer.csv')
print(realSoccer.head(10))
print(realSoccer.columns)
#join fake and real soccer data, make real = 1, fake =0
fakeSoccer['real'] = 0
realSoccer['real'] = 1
soccer = pd.concat([fakeSoccer, realSoccer])
#soccer = soccer.sample(frac=1).reset_index(drop=True) #shuffle data to test concat
print(soccer.head(10))




