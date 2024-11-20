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
print(twitter.columns)

#read in albanian data
albanian = pd.read_csv('alb-fake-news-corpus.csv')
print(albanian.head(10))
print(albanian.columns)


#read and clean soccer data
fakeSoccer = pd.read_csv('fake-soccer.csv')
realSoccer = pd.read_csv('real-soccer.csv')
fakeSoccer['real'] = 0
realSoccer['real'] = 1
soccer = pd.concat([fakeSoccer, realSoccer])
text_column = 'tweet'  
soccer[text_column] = soccer[text_column].fillna("").astype(str) # Fill NaNs and convert to string fixes errors
print(soccer.head(10))

#start naive bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
from collections import Counter


#tokenize data
def naive_bayes_classifier(data, content_col, label_col, test_size=0.2, random_state=42, top_n=10):
    """
    Reusable function for Naive Bayes classification
    
    Args:
    data: pandas DataFrame
    content_col: column name containing text data (content for Albaian data)
    label_col: column name containing labels (fake_news for Albanian data)
    test_size: proportion of data to use for testing
    """

    # tokenize data
    def tokenize(text):
        return re.findall(r'\b\w+\b|[!?.,;]', text.lower())  # Tokenize words and punctuation

    data['tokens'] = data[content_col].apply(tokenize)

    # real/fake tokens
    fake_tokens = data[data[label_col] == 1]['tokens'].explode()
    real_tokens = data[data[label_col] == 0]['tokens'].explode()
    fake_word_counts = Counter(fake_tokens)
    real_word_counts = Counter(real_tokens)
    total_fake_words = sum(fake_word_counts.values())
    total_real_words = sum(real_word_counts.values())

    vocab = set(fake_word_counts.keys()).union(set(real_word_counts.keys()))
    vocab_size = len(vocab)

    word_likelihoods = {
        word: {
            'fake': (fake_word_counts[word] + 1) / (total_fake_words + vocab_size),
            'real': (real_word_counts[word] + 1) / (total_real_words + vocab_size)
        }
        for word in vocab
    }

    #nb prediction
    def predict_nb(content, likelihoods, prior_fake, prior_real):
        tokens = tokenize(content)
        log_prob_fake = np.log(prior_fake)
        log_prob_real = np.log(prior_real)

        for token in tokens:
            if token in likelihoods:
                log_prob_fake += np.log(likelihoods[token]['fake'])
                log_prob_real += np.log(likelihoods[token]['real'])

        return 1 if log_prob_fake > log_prob_real else 0

    # run classifier
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    prior_fake = train[label_col].mean()
    prior_real = 1 - prior_fake

    test['predicted'] = test[content_col].apply(
        predict_nb, args=(word_likelihoods, prior_fake, prior_real)
    )

    accuracy = accuracy_score(test[label_col], test['predicted'])
    print(f"Accuracy: {accuracy:.2f}")

    # print top words influencing fake news
    word_influence = {
        word: np.log(likelihood['fake']) - np.log(likelihood['real'])
        for word, likelihood in word_likelihoods.items()
    }

    sorted_words = sorted(word_influence.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:top_n]

    print("Top words pulling toward fake news:")
    for word, influence in top_words:
        print(f"Word: {word}, Pull: {influence:.4f}")

    return accuracy, top_words

#albanian data
X = albanian['content']
y = albanian['fake_news']
naive_bayes_classifier(albanian, 'content', 'fake_news')
naive_bayes_classifier(soccer, 'tweet', 'real')
