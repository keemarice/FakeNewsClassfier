from gensim.models import KeyedVectors

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#read in data (NEED TO TALK ABT THIS)
twitter = pd.read_csv('FakeNewsNet.csv') #ONLY INCLUDES TITLES OF ARTICLES
print(twitter.head(10)) 
print(twitter.columns)

#read in albanian data
albanian = pd.read_csv('alb-fake-news-corpus.csv')
print(albanian.head(10))
print(albanian.columns)

#read in soccer data
fakeSoccer = pd.read_csv('fake-soccer.csv')
realSoccer = pd.read_csv('real-soccer.csv')
fakeSoccer['real'] = 0
realSoccer['real'] = 1
soccer = pd.concat([fakeSoccer, realSoccer])
soccer['tweet'] = soccer['tweet'].fillna("").astype(str) # Fill NaNs and convert to string fixes errors
print(soccer.head(10))


word2vec_model = KeyedVectors.load_word2vec_format('/Users/karsi/Documents/GoogleNews-vectors-negative300.bin', binary=True)

def document_vector(doc, model):
    """Create a document vector by averaging Word2Vec embeddings of its words."""
    tokens = doc.split()  # Split text into tokens
    embeddings = [model[word] for word in tokens if word in model.key_to_index]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size) 

def naive_bayes_classifier_w2v(data, content_col, label_col, word2vec_model, test_size=0.2, random_state=42):
    """
    Naive Bayes classifier using Word2Vec embeddings.
    """
    data['doc_vector'] = data[content_col].apply(lambda x: document_vector(x, word2vec_model))
    X_train, X_test, y_train, y_test = train_test_split(
        np.stack(data['doc_vector'].values),  # Convert to 2D array
        data[label_col],
        test_size=test_size,
        random_state=random_state
    )

    # Train a Naive Bayes classifier
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Predict and evaluate
    from sklearn.metrics import accuracy_score
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes (Word2Vec) Accuracy: {accuracy:.2f} for Dataset: {data}")
    return accuracy

def knn_classifier_w2v(data, content_col, label_col, word2vec_model, test_size=0.2, random_state=42):
    """
    K-Nearest Neighbors classifier using Word2Vec embeddings.
    """
    data['doc_vector'] = data[content_col].apply(lambda x: document_vector(x, word2vec_model))

    X_train, X_test, y_train, y_test = train_test_split(
        np.stack(data['doc_vector'].values),  # Convert to 2D array
        data[label_col],
        test_size=test_size,
        random_state=random_state
    )

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"K-Nearest Neighbors (Word2Vec) Accuracy: {accuracy:.2f} for Dataset: {data}")
    return accuracy

def decision_tree_classifier_w2v(data, content_col, label_col, word2vec_model, test_size=0.2, random_state=42):
    """
    Decision Tree classifier using Word2Vec embeddings.
    """
    # Generate document vectors
    data['doc_vector'] = data[content_col].apply(lambda x: document_vector(x, word2vec_model))

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        np.stack(data['doc_vector'].values),  # Convert to 2D array
        data[label_col],
        test_size=test_size,
        random_state=random_state
    )

    # Train a Decision Tree classifier
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # Predict and evaluate
    from sklearn.metrics import accuracy_score
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree (Word2Vec) Accuracy: {accuracy:.2f} for Dataset: {data}")
    return accuracy

# Test the function
#naive_bayes_classifier_w2v(albanian, 'content', 'fake_news', word2vec_model)
#naive_bayes_classifier_w2v(soccer, 'tweet', 'real', word2vec_model)

knn_classifier_w2v(albanian, 'content', 'fake_news', word2vec_model)
knn_classifier_w2v(soccer, 'tweet', 'real', word2vec_model)

#decision_tree_classifier_w2v(albanian, 'content', 'fake_news', word2vec_model)
#decision_tree_classifier_w2v(soccer, 'tweet', 'real', word2vec_model)
