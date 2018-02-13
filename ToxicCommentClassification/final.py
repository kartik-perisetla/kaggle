import os
import csv
import sys
import nltk
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
import copy

CHAR_REP = 10

# to hold vectorizer and feature collection
cache = {}

def add_to_cache(key, value):
    if key in cache:
       print("cache already contains entry with key: '" + key + "'")
       raise Exception("Duplicate key in cache: cache already contains entry with key: '" + key + "'")
    else:
        cache[key] = value

def get_from_cache(key):
    if key in cache:
        return cache[key]
    else:
        print("cache doesn't contains entry with key '" + key + "'")
        raise Exception("Invalid cache access: cache doesn't contain key: '" + key + "'")


def change_to_curdir():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    # print("cur dir name:"+dname)
    os.chdir(dname)

def read_file(filepath, encoding='utf-8'):
    data = []
    with open(filepath,'rt',encoding=encoding) as f_input:
        data = list(csv.reader(f_input))[1:]
    f_input.close()
    print("read file '" + filepath + "' completely")
    return data


def get_count_dict(text):
    tokens = nltk.word_tokenize(text)
    result = Counter(tokens)
    return dict(result)

def get_binary_dict(text, tolerance=0):
    count_dict = get_count_dict(text)
    return {key:1 if val > tolerance else 0 for key,val in count_dict.items()}

# TODO: implement - get tf dict for input text
def get_term_freq_dict(text, tolerance=0):
    pass

def extract_features(list_of_text, vectorizer= DictVectorizer()):
    return vectorizer.fit_transform([get_binary_dict(text) for text in list_of_text]), vectorizer

def extract_test_features(list_of_text, vectorizer):
    return vectorizer.transform([get_binary_dict(text) for text in list_of_text])
    
def train_on_features(class_label, train_features, train_labels, model_skeleton):
    model = copy.copy(model_skeleton)
    print("about to fit model:'"+model.estimator.__class__.__name__+"' on " + str(len(train_labels)) + " instances")
    model.fit(train_features, train_labels)
    print("fitting complete on model:'"+model.estimator.__class__.__name__+"' on " + str(len(train_labels)) + " instances")

    print("\n")
    print("*"*CHAR_REP + model.estimator.__class__.__name__ + "*"*CHAR_REP )
    # print model params
    print("\nBest params for '"+ model.estimator.__class__.__name__ + "' for class '" + class_label + "' => " + str(model.best_params_))

    # commenting -> no need now as we are doing parameter sweep
    # print("Cross-Validation Accuracy for '"+ model.estimator.__class__.__name__ + "' for class '" + class_label + "' => ")
    # print(cross_validate(model, train_features, train_labels))

    print("*"*20 + "\n")
    return model

# pass 'TRAIN' as key when want features for training instances
# pass 'TEST' as key when want features for test instances
# pass None as key when adctually want to invoke feature extraction method
def get_features(text_list, key=None):  
    if key is None:
        return extract_features(text_list)
    else:
        train_features, vectorizer = get_from_cache(key)
        return (train_features, vectorizer)

def train_on_file(train_file_name, model_skeleton):
    train_data = read_file(train_file_name)
    text_list = [dat[1] for dat in train_data]
    labels = [dat[2:] for dat in train_data]
    # print(labels[:5])
    
    
    print("about to extract features")
    train_features, vectorizer = extract_features(text_list)
    print("extracted features")
    #pickle.dump(train_features, open("features.pkl","w"))

    model_collection = []
    types = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    for idx in range(len(labels[0])):        
        model_collection.append(train_on_features(types[idx], train_features, [x[idx] for x in labels], model_skeleton))
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

    
    models = []
    scores = ['precision', 'recall', 'accuracy']

    nb_params = [{"alpha" : [0.25, 0.5, 0.75, 1]}]

    svm_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                 ]

    lr_params = [
                    {
                    "penalty" : ["l1", "l2"],
                    "C": [0.25, 0.5, 0.75, 1]
                    }             
                ]

    #TODO : use term-frequency dict vectorizer and try all these algorithms

    for score in scores:
        models = [
                    GridSearchCV(estimator=BernoulliNB(), param_grid=nb_params , cv=5, scoring='%s_macro' % score, verbose=True),
                    GridSearchCV(estimator=svm.SVC(), param_grid=svm_params, cv=5, scoring='%s_macro' % score, verbose=True),
                    GridSearchCV(estimator=LogisticRegression(dual = True, tol=1e-6, class_weight='balanced'), param_grid=lr_params, cv=5, scoring='%s_macro' % score, verbose=True)
                ]
    


        for model_skeleton in models:
            print("Training '" + model_skeleton.estimator.__class__.__name__ + "' on file")
            
            # print(args)

            model_collection, vectorizer = train_on_file(args[0], model_skeleton)
            print("Testing on file")
            ids, results = test_on_file(args[1], model_collection, vectorizer)
            print("Writing to file")
            write_to_submission_file(model_skeleton.estimator.__class__.__name__ + "_" + args[2],ids,results)
    

if __name__ == '__main__':
    change_to_curdir()
    if len(sys.argv) >= 4:
        print("launched with args:" + str(sys.argv))
        run(sys.argv[1:])
    else:
        print("Must pass atleast 3 params.")
        print("python final.py <train_file_name> <test_file_name> <output_file_name>")