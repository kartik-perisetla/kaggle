import os
import csv
import sys
import nltk
import pickle
import copy
from ast import literal_eval
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

CHAR_REP = 10
TRAIN_KEY = 'TRAIN'
TEST_KEY = 'TEST'
BINARY_VECTORIZER_KEY = 'BINARY_VECTORIZER'
COUNT_VECTORIZER_KEY = 'COUNT_VECTORIZER'
CLASS_LABELS_KEY = 'CLASS_LABELS'
BEST_MODEL_COLLECTION_KEY = "best_model_collection.bin"
VECTORIZER_KEY = "binary_vectorizer.bin"

# to hold vectorizer and feature collection
cache = {}
best_model_dict = {}

def add_to_cache(key, value):
    if key in cache:
       print("cache already contains entry with key: '" + key + "'")
       raise Exception("Duplicate key in cache: cache already contains entry with key: '" + key + "'")
    else:
        cache[key] = value
        print("Added key-value pair with key:'" + key +"'")

def get_from_cache(key):
    if key in cache:
        return cache[key]
    else:
        print("cache doesn't contains entry with key '" + key + "'")
        return None


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
    print("Cross-Validation Accuracy for '"+ model.estimator.__class__.__name__ + "' for class '" + class_label + "' => ")

    cross_validation_score = cross_validate(model, train_features, train_labels)
    print(cross_validation_score)

    # Save the best model for the class label based on CV accuracy
    best_model_accuracy = best_model_dict.get(class_label, (0, None, None))
    if best_model_accuracy[0] < cross_validation_score:
        best_model_dict[class_label] = (cross_validation_score, model, )

    print("*"*20 + "\n")
    return model


def get_features(text_list, key=None):
    """Method to get features for input list of text
    
    Arguments:
        text_list {string} -- list of strings for which we need to extract features
    
    Keyword Arguments:
        key {string} -- string to identify instances corresponding to a logical group. i.e. TRAIN or TEST set (default: {None})
        - pass 'TRAIN_KEY' as key when want features for training instances
        - pass 'TEST_KEY' as key when want features for test instances
        - pass None as key when actually want to invoke feature extraction method
    """    
    if key is None:
        train_features, vectorizer = extract_features(text_list)
        add_to_cache(BINARY_VECTORIZER_KEY, vectorizer)
        return (train_features, vectorizer)
    else:
        item = get_from_cache(key)
        # if item is a tuple of train_features and vectorizer
        if item is not None:
            train_features, vectorizer = item
        else:
            # extract features using input text_list
            train_features, vectorizer = extract_features(text_list)

            # cache the vectorizer and train features for dataset identified with a key
            add_to_cache(BINARY_VECTORIZER_KEY, vectorizer)
            add_to_cache(key, (train_features, vectorizer))
        return (train_features, vectorizer)

# method that trains a collection of model
def train_on_file(train_file_name, model_skeleton):
    train_data = read_file(train_file_name)
    text_list = [dat[1] for dat in train_data]
    labels = [dat[2:] for dat in train_data]
    
    
    print("about to extract features")
    # train_features, vectorizer = extract_features(text_list)

    train_features, vectorizer = get_features(text_list, TRAIN_KEY)

    print("extracted features")

    # dumping features to file
    #pickle.dump(train_features, open("features.pkl","w"))

    model_collection = []
    class_labels = get_from_cache(CLASS_LABELS_KEY)

    # training 
    for idx in range(len(labels[0])):
        # given the model_skeleton, find the best params given input features for train instances
        model_collection.append(train_on_features(class_labels[idx], train_features, [x[idx] for x in labels], model_skeleton))
    return model_collection, vectorizer

def predict(test_features, model):
    if hasattr(model, "predict_proba"):
        result = model.predict_proba(test_features)
        print(model.classes_)
        #one_index = model.classes_.index("1")
        one_index = 1
        probabilities = [item[one_index] for item in result]
    else:
        probabilities = model.predict(test_features)
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
    writo = [get_from_cache(CLASS_LABELS_KEY)]
    with open(filepath, "wt", encoding="utf-8", newline='') as fop:
        writer = csv.writer(fop)       
        for idx, item in zip(ids, results):
            writo.append([idx] + list(item))
        writer.writerows(writo)

def cross_validate(model, feature_set, true_labels):
    predicted = cross_val_predict(model, feature_set, true_labels, cv=10)
    return metrics.accuracy_score(true_labels, predicted)


def pickle_object(instance, pickle_name="pickled_object.bin"):
    if instance is not None:
        pickle.dump(instance, open(pickle_name, "wb"))
        print("pickle_object:instance pickled as '" + pickle_name + "' successfully.")
    else:
        print("pickle_object:make sure instance is not None.")

def unpickle_object(pickle_name="pickled_object.bin"):
    if not os.path.exists(pickle_name):
        print("unpickled_object:pickle file '" + pickle_name +"' not found :(")
        return
    instance = pickle.load(open(pickle_name, "rb"))
    return instance

def save_model_collection(model_collection, pickle_name="model_collection.bin"):
    pickle_object(model_collection, pickle_name)

def load_model_collection(pickle_name="model_collection.bin"):
    return unpickle_object(pickle_name=pickle_name)

def save_vectorizer(vectorizer, pickle_name="vectorizer.bin"):
    pickle_object(vectorizer, pickle_name)

def load_vectorizer(pickle_name="vectorizer.bin"):
    return unpickle_object(pickle_name=pickle_name)


# args : <train_file_name> <test_file_name> <output_file_name> <skip_train:True/False>
def run(args):

    class_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    add_to_cache(CLASS_LABELS_KEY, class_labels)

    models = []
    scores = ['precision', 'recall', 'accuracy']

    skip_train_flag = literal_eval(args[3])

    if not skip_train_flag:
        nb_params = [{"alpha" : [0.25, 0.5, 0.75, 1]}]    

        svm_params = [
                        {
                        "C" : [0.1, 0.25, 0.5, 0.75, 1]
                        }
                    ]

        linear_params = [
                            {
                                "fit_intercept" : [True, False]
                            }

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
                        # GridSearchCV(estimator=LinearRegression(), param_grid=linear_params, cv=5, scoring='%s_macro' % score, verbose=True),

                        GridSearchCV(estimator=LogisticRegression(dual = False, class_weight='balanced'), param_grid=lr_params, cv=5, scoring='%s_macro' % score, verbose=True),

                        GridSearchCV(estimator=svm.LinearSVC(), param_grid=svm_params, cv=5, scoring='%s_macro' % score, verbose=True),
                        
                        GridSearchCV(estimator=BernoulliNB(), param_grid=nb_params , cv=5, scoring='%s_macro' % score, verbose=True)
                    ]

            for model_skeleton in models:
                print("Training '" + model_skeleton.estimator.__class__.__name__ + "' on file")
                model_collection, vectorizer = train_on_file(args[0], model_skeleton)
        
        best_model_collection = []
        # pick the best models for each class - these were captured in best_model_dict in train_on_features method
        for class_label in get_from_cache(CLASS_LABELS_KEY):
            best_model_collection.append(best_model_dict[class_label][1])

        # save the best_model_collection
        save_model_collection(model_collection=best_model_collection, pickle_name=BEST_MODEL_COLLECTION_KEY)

        #save the vectorizer
        binary_vectorizer = get_from_cache(BINARY_VECTORIZER_KEY)
        save_vectorizer(vectorizer=binary_vectorizer, pickle_name=VECTORIZER_KEY)        
    else:
        # load the pickled trained best model collection
        best_model_collection = load_model_collection(pickle_name=BEST_MODEL_COLLECTION_KEY)
        binary_vectorizer = load_vectorizer(pickle_name=VECTORIZER_KEY)
        print("loaded pre-trained best_collection_model successfully.")

    print("*"*10 + "Testing on file" + "*"*10)
    ids, results = test_on_file(args[1], best_model_collection, binary_vectorizer)
    print("Writing to file")
    write_to_submission_file(args[2],ids,results)
    

if __name__ == '__main__':
    change_to_curdir()
    if len(sys.argv) >= 4:
        print("launched with args:" + str(sys.argv))
        run(sys.argv[1:])
    else:
        print("Must pass atleast 3 params.")
        print("python final.py <train_file_name> <test_file_name> <output_file_name>")