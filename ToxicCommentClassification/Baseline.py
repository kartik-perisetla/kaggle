import csv
import sys
import nltk
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import copy

def read_file(filePath, encoding='utf-8'):
    data = []
    with open(sys.argv[2],'rt',encoding=encoding) as f_input:
        data = list(csv.reader(f_output))[1:]
    return data


def get_count_dict(text):
    tokens = nltk.word_tokenize(text)
    result = Counter(tokens)
    return dict(result)

def get_binary_dict(text, tolerance=0):
    count_dict = get_count_dict(text)
    return {key:1 if val > tolerance else key:0 for key,val in count_dict.items()}

def extract_features(list_of_text, vectorizer= DictVectorizer(sparse=False)):
    return vectorizer.fit_transform([get_binary_text(text) for text in list_of_text])
    
def train_on_features(train_features, train_labels, model_skeleton):
    model = copy.copy(model_skeleton)
    model.fit(train_features, train_labels)
    return model
    

def train_on_file(train_file_name, model_skeleton):
    train_data = read_file(train_file_name)
    text_list = [dat[1] for dat in train_data]
    labels = [dat[2:] for dat in labels]
    
    train_features = extract_features(text_list)

    model_collection = []
    for idx in range(len(labels[0])):
        model_collection.append(train_on_features(train_features, [x[idx] for x in labels], model_skeleton))
    return model_collection

def predict(test_features, model):
    result = model.predict_proba(test_features)
    print(model.classes_)
    one_index = model.classes_.index("1")
    probabilities = [item[one_index] for item in result]
    return probabilities


def test_on_file(test_file_name, model_collection):
    test_data = read_file(test_file_name)
    ids = [dat[0] for dat in test_data]
    text_list = [dat[1] for dat in test_data]
    test_features = extract_features(text_list)
    test_probabilities = []
    for model in model_collection:
        model_probabilities = predict(test_features, model)
        test_probabilities.append(model_probabilities)
    return ids, list(zip(*test_probabilities))
    
def write_to_submission_file(filepath, ids, results):
    writo = [['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    with open(filepath, "w", encoding="utf-8") as fop:
        writer = csv.writer(fop,newline='')       
        for idx, item in zip(ids, results):
            writo.append([idx] + list(item))
        writer.writerows(writo)
        
def run(args):
    model_skeleton = ""
    model = train_on_file(args[0], model_skeleton)
    ids, results = test_on_file(args[1], model)
    write_to_submission_file(args[2],ids,results)
    

if __name__ == '__main__':
    run(sys.argv[1:])
