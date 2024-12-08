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
soccer['tweet'] = soccer['tweet'].fillna("").astype(str) # Fill NaNs and convert to string fixes errors
print(soccer.head(10))



#start naive bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
from collections import Counter
from sklearn.metrics import classification_report


def tokenize(text):
    return re.findall(r'\b\w+\b|[!?.,;]', text.lower())  

def naive_bayes_classifier_count(data, content_col, label_col, test_size=0.2, random_state=42, top_n=10):
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

    #laplace smoothing for word likelihoods
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

        return 1 if log_prob_fake > log_prob_real else 0 # 1 is fake, 0 is real

    # run classifier
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    prior_fake = train[label_col].mean()
    prior_real = 1 - prior_fake

    test['predicted'] = test[content_col].apply(
        predict_nb, args=(word_likelihoods, prior_fake, prior_real)
    )

    accuracy = accuracy_score(test[label_col], test['predicted'])
    print(f"NB (w/ laplace smoothing Accuracy: {accuracy}")

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

    #print classification report
    print("\nClassification Report:")
    print(classification_report(test[label_col], test['predicted']))
    
    # Plot misclassified 
    misclassified = test[test[label_col] != test['predicted']]
    plt.figure(figsize=(10, 6))
    plt.hist(misclassified[content_col].apply(len), bins=20, color='red', alpha=0.7, label='Misclassified')
    plt.hist(test[content_col].apply(len), bins=20, color='blue', alpha=0.5, label='All')
    plt.xlabel('Length of Content')
    plt.ylabel('Frequency')
    plt.title('Distribution of Content Lengths for Misclassified Examples')
    plt.legend()
    plt.show()
    
  
    return accuracy, top_words

def naive_bayes_classifier_tfidf(data, content_col, label_col, test_size=0.2, random_state=42, top_n=10, alpha=1.0):
    # Preprocess text
    data['tokens'] = data[content_col].apply(lambda x: ' '.join(tokenize(x)))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['tokens'], data[label_col], test_size=test_size, random_state=random_state
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb_model = MultinomialNB(alpha=alpha)
    nb_model.fit(X_train_vec, y_train)

    y_pred = nb_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Accuracy (with alpha={alpha}): {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_names = vectorizer.get_feature_names_out()
    feature_log_probs = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
    top_indices = feature_log_probs.argsort()[::-1][:top_n]
    top_words = [(feature_names[i], feature_log_probs[i]) for i in top_indices]

    print(f"\nTop {top_n} Influential Words:")
    for word, score in top_words:
        print(f"Word: {word}, Score: {score:.4f}")

    #plot misclassified
    misclassified = data.loc[y_test[y_test != y_pred].index]
    plt.figure(figsize=(10, 6))
    plt.hist(misclassified[content_col].apply(len), bins=20, color='red', alpha=0.7, label='Misclassified')
    plt.hist(data[content_col].apply(len), bins=20, color='blue', alpha=0.5, label='All')
    plt.xlabel('Length of Content')
    plt.ylabel('Frequency')
    plt.title('Distribution of Content Lengths for Misclassified Examples')
    plt.legend()
    plt.show()
    
    return accuracy, top_words



#start knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

#CURSE OF DIMENTIONALITY
def knn_classifier(data, content_col, label_col, test_size=0.2, random_state=42, n_neighbors=10, top_n=10):
    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))
    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)

    # vectorize text data using TF-IDF useful for KNN 
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # get influential features
    feature_names = vectorizer.get_feature_names_out()
    feature_scores = X_train.mean(axis=0).A1  
    top_indices = feature_scores.argsort()[::-1][:top_n]  
    top_features = [(feature_names[i], feature_scores[i]) for i in top_indices]

    print(f"\nTop {top_n} Influential Features:")
    for feature, score in top_features:
        print(f"Feature: {feature}, Score: {score:.4f}")

    #plot 
    plt.figure(figsize=(10, 6))
    words, scores = zip(*top_features)
    plt.barh(words, scores, color='skyblue', edgecolor='black')
    plt.gca().invert_yaxis()  # Invert y-axis for descending order
    plt.title(f"Top {top_n} Influential Features for KNN Classification")
    plt.xlabel("Feature Scores (TF-IDF Mean)")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.show()

    return accuracy, top_features

#start svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def svm_classifier(data, content_col, label_col, test_size=0.2, random_state=42):
    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))
    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)

    # vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # standardize features
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train SVM
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.2f}")

    #print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix for SVM')
    plt.show()

    return accuracy

from sklearn.tree import DecisionTreeClassifier, export_text
def decision_tree_classifier(data, content_col, label_col, test_size=0.2, random_state=42, top_n=10, max_depth_range=(1, 50)):
    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    best_accuracy = 0
    best_max_depth = None
    best_top_features = None
    

    for max_depth in range(max_depth_range[0], max_depth_range[1] + 1):
        dt = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_max_depth = max_depth

    print(f"\nBest Max Depth: {best_max_depth}")
    print(f"Best Accuracy: {best_accuracy:.2f}")

    print(f"\nTop {top_n} Influential Features:")

    from sklearn.tree import plot_tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=vectorizer.get_feature_names_out(), max_depth=2, filled=True)
    plt.show()

    
    return best_accuracy, best_max_depth, best_top_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def random_forest_classifier(data, content_col, label_col, test_size=0.2, random_state=42, top_n=10, n_estimators=100):
   
    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    accuracy_list = []  # Store accuracy for each number of estimators
    best_accuracy = 0
    best_n_estimators = None
    feature_importances = None

    for n in range(1, n_estimators + 1):
        rf = RandomForestClassifier(random_state=random_state, n_estimators=n)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_estimators = n
            feature_importances = rf.feature_importances_  

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_estimators + 1), accuracy_list, marker='o', color='blue', label='Accuracy')
    plt.axvline(x=best_n_estimators, color='red', linestyle='--', label=f'Best Estimators: {best_n_estimators}')
    plt.title('Accuracy vs. Number of Estimators (Random Forest)')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    if feature_importances is not None:
        feature_names = vectorizer.get_feature_names_out()
        top_indices = feature_importances.argsort()[::-1][:top_n]
        top_features = [(feature_names[i], feature_importances[i]) for i in top_indices]
        
        # Bar chart for top features
        plt.figure(figsize=(8, 6))
        feature_names_plot, importance_scores = zip(*top_features)
        plt.barh(feature_names_plot, importance_scores, color='green', alpha=0.7)
        plt.title(f'Top {top_n} Influential Features')
        plt.xlabel('Feature Importance')
        plt.gca().invert_yaxis()  
        plt.tight_layout()
        plt.show()
    else:
        top_features = None

    print(f"\nBest Number of Estimators: {best_n_estimators}")
    print(f"Best Accuracy: {best_accuracy:.2f}")

    return best_accuracy, best_n_estimators, top_features


from sklearn.linear_model import LogisticRegression

def logistic_regression_classifier(data, content_col, label_col, test_size=0.2, random_state=42):
    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))
    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")

    return accuracy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def lda_classifier(data, content_col, label_col, test_size=0.2, random_state=42):

    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()  
    X_test = vectorizer.transform(X_test).toarray()       

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LDA Accuracy: {accuracy:.2f}")

    return accuracy

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
def qda_classifier(data, content_col, label_col, test_size=0.2, random_state=42):

    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))
    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()  
    X_test = vectorizer.transform(X_test).toarray()   
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    y_pred = qda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"QDA Accuracy: {accuracy:.2f}")

    return accuracy

from sklearn.neural_network import MLPClassifier
def neural_network_classifier(data, content_col, label_col, test_size=0.2, random_state=42):
    """
    Reusable function for neural network classification
    
    Args:
    data: pandas DataFrame
    content_col: column name containing text data
    label_col: column name containing labels
    test_size: proportion of data to use for testing

    Returns:
    accuracy: accuracy of classifier
    """

    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)

    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()  # Convert sparse matrix to dense
    X_test = vectorizer.transform(X_test).toarray()        # Convert sparse matrix to dense

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train neural network
    nn = MLPClassifier()
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Neural Network Accuracy: {accuracy:.2f}")


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

def hybrid_knn_naive_bayes(data, content_col, label_col, test_size=0.2, random_state=42, weight_nb=0.6, weight_knn=0.4, n_neighbors=5):
    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))
    X_train, X_test, y_train, y_test = train_test_split(
        data['tokens'], data[label_col], test_size=test_size, random_state=random_state
    )
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    #NB
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vectorized, y_train)
    nb_probs = nb_model.predict_proba(X_test_vectorized)

    #KNN
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train_vectorized, y_train)
    knn_probs = knn_model.predict_proba(X_test_vectorized)

    # Combine predictions using weighted voting
    combined_probs = (weight_nb * nb_probs) + (weight_knn * knn_probs)
    y_pred = combined_probs.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Hybrid Model Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    def plot_model_contributions(weight_nb, weight_knn):
        plt.figure(figsize=(6, 5))
        plt.bar(['Naive Bayes', 'KNN'], [weight_nb, weight_knn], color=['skyblue', 'lightgreen'])
        plt.ylabel('Weight Contribution')
        plt.title('Hybrid Model Contributions')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    plot_model_contributions(weight_nb, weight_knn)

    return accuracy

def hybrid_nb_lr(data, content_col, label_col, test_size=0.2, random_state=42):
    """
    Hybrid model combining Naive Bayes and Logistic Regression.
    """
    data['tokens'] = data[content_col].apply(lambda x: ' '.join(tokenize(x)))
    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)
    
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # nb
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_train_probs = nb.predict_proba(X_train_vec)
    nb_test_probs = nb.predict_proba(X_test_vec)
    
    # logistic on nb
    lr = LogisticRegression()
    lr.fit(nb_train_probs, y_train)
    y_pred = lr.predict(nb_test_probs)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Hybrid NB + LR Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def neural_network_classifier(data, content_col, label_col, test_size=0.2, random_state=42):

    data['tokens'] = data[content_col].apply(tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

    X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data[label_col], test_size=test_size, random_state=random_state)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()  
    X_test = vectorizer.transform(X_test).toarray()        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # use MLP from sklearn
    nn = MLPClassifier()
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Neural Network Accuracy: {accuracy}")

    return accuracy

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def neural_network_classifier_tensorflow(data, content_col, label_col, test_size=0.2, random_state=42, max_words=5000, max_len=100):
    # Tokenize text data not using TF-IDF, instead using Tokenizer from Keras
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(data[content_col])
    sequences = tokenizer.texts_to_sequences(data[content_col])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data[label_col], test_size=test_size, random_state=random_state)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(max_len,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # For binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network Accuracy: {test_accuracy}")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return test_accuracy, history

def plot_training_history(history, model_name='Neural Network'):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


#run classifiers


#print accuracies
accuracies = {}
accuracies['albanian'] = {}
accuracies['soccer'] = {}
accuracies['twitter'] = {}

accuracies['albanian']['naive_bayes'] = naive_bayes_classifier_count(albanian, 'content', 'fake_news')
accuracies['albanian']['knn'] = knn_classifier(albanian, 'content', 'fake_news')
accuracies['albanian']['svm'] = svm_classifier(albanian, 'content', 'fake_news')
accuracies['albanian']['decision_tree'] = decision_tree_classifier(albanian, 'content', 'fake_news')
accuracies['albanian']['random_forest'] = random_forest_classifier(albanian, 'content', 'fake_news')
accuracies['albanian']['logistic_regression'] = logistic_regression_classifier(albanian, 'content', 'fake_news')
accuracies['albanian']['knn/naive_bayes'] = hybrid_knn_naive_bayes(albanian, 'content', 'fake_news')
accuracies['albanian']['nb/lr'] = hybrid_nb_lr(albanian, 'content', 'fake_news')

accuracies['soccer']['naive_bayes'] = naive_bayes_classifier_count(soccer, 'tweet', 'real')
accuracies['soccer']['knn'] = knn_classifier(soccer, 'tweet', 'real')
#accuracies['soccer']['svm'] = svm_classifier(soccer, 'tweet', 'real')
#accuracies['soccer']['decision_tree'] = decision_tree_classifier(soccer, 'tweet', 'real')
#accuracies['soccer']['random_forest'] = random_forest_classifier(soccer, 'tweet', 'real')
accuracies['soccer']['logistic_regression'] = logistic_regression_classifier(soccer, 'tweet', 'real')
accuracies['soccer']['knn/naive_bayes'] = hybrid_knn_naive_bayes(soccer, 'tweet', 'real')
accuracies['soccer']['nb/lr'] = hybrid_nb_lr(soccer, 'tweet', 'real')


soccer_small = soccer.sample(1000)
accuracies['soccer']['svm'] = svm_classifier(soccer_small, 'tweet', 'real')
accuracies['soccer']['decision_tree'] = decision_tree_classifier(soccer_small, 'tweet', 'real')
accuracies['soccer']['random_forest'] = random_forest_classifier(soccer_small, 'tweet', 'real')

#test neural nets
accuracy_nn = neural_network_classifier(albanian, 'content', 'fake_news')
accuracy_tf, history_tf = neural_network_classifier_tensorflow(albanian, 'content', 'fake_news')

plot_training_history(history_tf, model_name='Neural Network - TensorFlow (Albanian)')



accuracies = pd.DataFrame(accuracies)
print(accuracies)

print(albanian.shape)
print(soccer.shape)




