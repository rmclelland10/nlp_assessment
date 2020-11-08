#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ross
"""

from oa_p1 import get_raw_text
from nltk.corpus import stopwords
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

STOP_WORDS = stopwords.words('english')

def clean_text(sentences):
    '''
    Parameters
    ----------
    sentences : list[str, str, ..]
        Sentence list to be cleaned.
    Returns
    -------
    sentences : list[str, str, ..]
        Cleaned sentence list.
    '''
    from nltk import PorterStemmer
    ps = PorterStemmer()
    for i in range(len(sentences)):
        words = re.findall('[a-z0-9\-]+[\-]?[a-z0-9%]+|[a-z0-9]|\([+-]\)', sentences[i])
        sentences[i] = ' '.join([ps.stem(word) for word in words])
        sentences[i] = sentences[i].replace('(-)', 'NEGATIVE_TEST_RESULT').replace('(+)', 'POSITIVE_TEST_RESULT')
        sentences[i] = sentences[i].replace('%', '_PERCENT').replace('-', '_')
        sentences[i] = ' '.join([word for word in sentences[i].split() if word not in STOP_WORDS]) # Remove stop-words
    return sentences

def get_medical_term_count(data):
    '''
    Parameters
    ----------
    data: DataFrame
        Dataframe containing text sentences
    Returns
    -------
    med_terms_per_label : list[float, float, ..]
        fraction of medical terms normalized across sentences
    med_terms_per_sentence : list[float, float, ..]
        # medical terms / total words in sentence, per sentence
    '''
    with open('medicalterms.txt', 'r') as file:
        med_terms = file.read()
        med_terms = med_terms.replace("'", '')
        med_terms = med_terms.lower()
        
    med_terms = set(med_terms.split('\n'))
    med_term_count_list = []
    med_terms_per_sentence = []
    sentences = data['sentences'].values
    for i in range(len(sentences)):
        count = 0
        for word in sentences[i].split():
            if word in med_terms:
                count += 1
        med_term_count_list.append(count)
        med_terms_per_sentence.append(count / len(sentences[i].split()))
    
    if max(med_term_count_list) > 0:
        med_terms_per_label = [(item - min(med_term_count_list)) / (max(med_term_count_list) - min(med_term_count_list)) for item in med_term_count_list]
    else:
        med_terms_per_label = med_term_count_list
        
    return med_terms_per_label, med_terms_per_sentence
        

def process_new_file_data(filename, classification=None, labels=None):
    '''
    Parameters
    ----------
    filename : str
        File/text document to be processed
    classification : int
        Type of document. Each sentence within document will be asigned the same type
        1 - General
        2 - Pathology Results
        3 - Lab Results
        4 - Imaging Results
        Cannot be used if labels is passed as argument.
    labels : list[int, int, ...]
        List of manually signed/verified types for each sentence (see types above).
        Cannot be used if classfication is passed as argument.
    Returns
    -------
    data - dataframe consisting of sentences and classification.

    '''
    if classification and labels:
        raise Exception('Error. Both classfication and labels were passed. Can only use 1.')
    elif not classification and not labels:
        raise Exception('Error. Neither classification or labels were passed.')
    
    from nltk import tokenize
    raw_text = get_raw_text(filename=filename)
    sentences = tokenize.sent_tokenize(raw_text)
    sentences = clean_text(sentences)
    
    data = pd.DataFrame()
    data['sentences'] = sentences
    med_term_stats = get_medical_term_count(data)
    data['medicalterms_per_label'] = med_term_stats[0]
    data['medicalterms_per_sentence'] = med_term_stats[1]
    
    if classification:
        data['labels'] = [classification] * len(sentences)
    elif labels:
        data['labels'] = labels
    return data

def neighbor_sentence_classifcation(data):
    '''
    Parameters
    ----------
    data: DataFrame
        Dataframe containing text features
    Returns
    data: DataFrame
        Dataframe with neighborbefore and neighborafter columns added.
        neighborbefore = 1 if sentence before is of the same class, else 0
        neighborafter = 1 if sentence after is of the same class, else 0
    -------
    '''
    label_list = data['labels'].values
    neighbors = []
    for i in range(len(label_list)):
        if i == 0:
            before = 0
            after = int(label_list[i] == label_list[i + 1])
        elif i == len(label_list) - 1:
            before = int(label_list[i] == label_list[i - 1])
            after = 0
        else:
            before = int(label_list[i] == label_list[i - 1])
            after = int(label_list[i] == label_list[i + 1])
        neighbors.append([before, after])
    data['neighborbefore'] = [item[0] for item in neighbors]
    data['neighborafter'] = [item[1] for item in neighbors]
    return data
        
def combine_features(data):
    # Create the Bag of Words model, set X and y for model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(list(data['sentences'].values)).toarray()
    
    # Combine all X features
    X_df = pd.DataFrame(X)
    X_df['medicalterms_per_label'] = data['medicalterms_per_label'].values
    X_df['medicalterms_per_sentence'] = data['medicalterms_per_sentence'].values
    X_df['neighborbefore'] = data['neighborbefore'].values
    X_df['neighborafter'] = data['neighborafter'].values
    X = X_df.values
    
    # set y data
    y = data['labels'].values

    return X, y
    

def create_model(X, y, test_size=0.25, classifier_obj=GaussianNB, show_plot=False, save_model=False):
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    # Training the Naive Bayes model on the Training set
    model = classifier_obj()
    model.fit(X_train, y_train)
    
    # Predicting the Test set results, calc confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot confusion matrix
    if show_plot:
        from helpers import plot_confusion_matrix
        plot_confusion_matrix(cm, classes = [1, 2, 3, 4])
        
    if save_model:
        from datetime import datetime
        dt= datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        filename = 'model-{}.sav'.format(dt)
        pickle.dump(model, open(filename, 'wb'))
        
    print('====== RESULTS FOR MODEL ======')
    print('y_test:\t{}'.format(y_test))
    print('y_pred:\t{}'.format(y_pred))
    print('accuracy\t{}'.format(accuracy))
    print('confusion matrix:\n{}'.format(cm))
    print('====== END OF RESULTS =========')
    
    return [model, y_pred, cm, accuracy] 


if __name__ == '__main__':
    classifiers = [LogisticRegression, GaussianNB, DecisionTreeClassifier, RandomForestClassifier, SVC]
    
    # PARAMETERS THAT NEED SET
    text_filename = 'exercise_text.txt' # file to be read
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4] # sentence labels/classification (1 - General, 2 - Pathology Results, 3 - Lab Results, 4 - Imaging Results)
    test_size = 0.25 # fraction of data to be used for testing
    classifier_obj_index = 0 # index of classifier object in classifiers list above
    show_plot = False # set to true to see confusion matrix plot
    save_model = True # set to true to save model as .av file
    # END OF PARAMETERS THAT NEED SET

    # Cleanse text file using inputted file and pre-set labels corresponding to classification of each sentence
    # DataFrame returns with clean sentences, medicalterms_per_label, medicalterms_per_sentence, and labels
    data = process_new_file_data(filename=text_filename, labels=labels)

    # Add columns for neighborbefore (sentence before has same class) and neighborafter (sentence after has same class)
    data = neighbor_sentence_classifcation(data)

    # Convert sentences to bag of words model, combine BOW array with other features
    X, y = combine_features(data)

    # Create model, get stats/display
    results = create_model(X, y, test_size=test_size, classifier_obj=classifiers[classifier_obj_index], show_plot=show_plot, save_model=save_model)
    
    
    


    














