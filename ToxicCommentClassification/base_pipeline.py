import csv
import sys
import nltk
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import pickle

import copy

def read_file(filepath, encoding='utf-8'):
    data = []
    with open(filepath,'rt',encoding=encoding) as f_input:
        data = list(csv.reader(f_input))[1:]
    return data


def get_count_dict(text):
    tokens = nltk.word_tokenize(text)
    result = Counter(tokens)
    return dict(result)

def get_binary_dict(text, tolerance=0):
    count_dict = get_count_dict(text)
    return {key:1 if val > tolerance else 0 for key,val in count_dict.items()}

def extract_features(list_of_text, vectorizer= DictVectorizer()):
    return vectorizer.fit_transform([get_binary_dict(text) for text in list_of_text]), vectorizer

def extract_test_features(list_of_text, vectorizer):
    return vectorizer.transform([get_binary_dict(text) for text in list_of_text])
    
def train_on_features(train_features, train_labels, model_skeleton):
    model = copy.copy(model_skeleton)
    model.fit(train_features, train_labels)

    print(cross_validate(model, train_features, train_labels))
    return model
    

def train_on_file(train_file_name, model_skeleton):
    train_data = read_file(train_file_name)
    text_list = [dat[1] for dat in train_data]
    labels = [dat[2:] for dat in train_data]
    print(labels[:5])
    
    train_features, vectorizer = extract_features(text_list)
    #pickle.dump(train_features, open("features.pkl","w"))

    model_collection = []
    types = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    for idx in range(len(labels[0])):
        print("Cross-Validation Accuracy for " + types[idx] + " : ")
        model_collection.append(train_on_features(train_features, [x[idx] for x in labels], model_skeleton))
    return model_collection, vectorizer

def predict(test_features, model):
    result = model.predict_proba(test_features)
    print(model.classes_)
    #one_index = model.classes_.index("1")
    one_index = 1
    probabilities = [item[one_index] for item in result]
    return probabilities


def test_on_file(test_file_name, model_collection, vectorizer):
    test_data = read_file(test_file_name)
    ids = [dat[0] for dat in test_data]
    text_list = [dat[1] for dat in test_data]
    test_features = extract_test_features(text_list, vectorizer)
    test_probabilities = []
    for model in model_collection:
        model_probabilities = predict(test_features, model)
        test_probabilities.append(model_probabilities)
    return ids, list(zip(*test_probabilities))
    
def write_to_submission_file(filepath, ids, results):
    writo = [['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    with open(filepath, "wt", encoding="utf-8", newline='') as fop:
        writer = csv.writer(fop)       
        for idx, item in zip(ids, results):
            writo.append([idx] + list(item))
        writer.writerows(writo)

def cross_validate(model, feature_set, true_labels):
    predicted = cross_val_predict(model, feature_set, true_labels, cv=10)
    return metrics.accuracy_score(true_labels, predicted)
    
def run(args):
    model_skeleton = BernoulliNB()
    print("Training on file")
    print(args)
    model_collection, vectorizer = train_on_file(args[0], model_skeleton)
    print("Testing on file")
    ids, results = test_on_file(args[1], model_collection, vectorizer)
    print("Writing to file")
    write_to_submission_file(args[2],ids,results)
    

if __name__ == '__main__':
    run(sys.argv[1:])
