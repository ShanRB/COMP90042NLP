"""
To generate result for Random Forest Classifier
This file will be runned using random forest, the result file wil be saved to
dev-output.json and test-output.json

Input   : train.json, negative_train.json, dev.json, test-unlabelled.json
Output  : dev-output.json, test-output.json
"""

import numpy as np
import json
import tokenizer
from get_features import get_features

from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import precision_recall_fscore_support

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def dummy(doc):
    return doc

if __name__ == '__main__':
    trainfilename = 'train.json'
    devfilename = 'dev.json'
    testfilename = 'test-unlabelled.json'

    trainfile = open(trainfilename)
    trainJson = json.load(trainfile)
    trainfile.close()

    devfile = open(devfilename)
    devJson = json.load(devfile)
    devfile.close()

    testfile = open(testfilename)
    testJson = json.load(testfile)
    testfile.close()

    negativefilename = 'negative_train.json'
    negativefile = open(negativefilename)
    nJson = json.load(negativefile)
    negativefile.close()
    
    # acquire features for training
    train_tokens,train_tags,train_otherfeats = get_features(trainJson)
    neg_tokens,neg_tags,neg_otherfeats = get_features(nJson)

    labels = tokenizer.getlabels(trainJson) + tokenizer.getlabels(nJson)
    train_tokens += neg_tokens
    train_tags += neg_tags
    train_otherfeats += neg_otherfeats

    #vec = TfidfVectorizer(tokenizer=dummy,preprocessor=dummy)
    vec = CountVectorizer(tokenizer=dummy,preprocessor=dummy)
    train_vec = vec.fit_transform(train_tokens).toarray()
    train_tags = csr_matrix(train_tags).toarray()
    train_otherfeats = csr_matrix(train_otherfeats).toarray()
    train_X = np.concatenate([train_otherfeats,train_tags,train_vec],axis=1)
    #train_X = train_vec
    print(train_X.shape)
    train_y = csr_matrix(labels).toarray()
    print(train_y.shape)

    # acquire dev features
    dev_labels = tokenizer.getlabels(devJson)
    dev_tokens,dev_tags,dev_otherfeats = get_features(devJson)
    dev_vec = vec.transform(dev_tokens).toarray()
    dev_tags = csr_matrix(dev_tags).toarray()
    dev_otherfeats = csr_matrix(dev_otherfeats).toarray()
    dev_X = np.concatenate([dev_otherfeats,dev_tags,dev_vec],axis=1)
    #dev_X = dev_vec

    # acquire test features
    test_tokens,test_tags,test_otherfeats = get_features(testJson)
    test_vec = vec.transform(test_tokens).toarray()
    test_tags = csr_matrix(test_tags).toarray()
    test_otherfeats = csr_matrix(test_otherfeats).toarray()
    test_X = np.concatenate([test_otherfeats,test_tags,test_vec],axis=1)

    
    # Random Forest
    
    # hyper parameter tunning
    nes_list = range(200,800,100)
    all_f = []
    all_p = []
    all_r = []
    for nes in nes_list:
        clf = RandomForestClassifier(n_estimators=nes,random_state=0)
        clf.fit(train_X,train_y.ravel())
        dev_y = clf.predict(dev_X)
        p, r, f, _ = precision_recall_fscore_support(\
             dev_labels, dev_y, pos_label=1, average="binary")
        all_f.append(f)
        all_p.append(p)
        all_r.append(r)
    best = max(all_f)
    best_nes = nes_list[all_f.index(best)]
    print(all_f)
    print(all_p)
    print(all_r)
    print(best, ' - ', best_nes)
    
    
    clf = RandomForestClassifier(n_estimators=best_nes,random_state=0)
    clf.fit(train_X,train_y.ravel())
    dev_y = clf.predict(dev_X)
    tokenizer.write_output('dev',dev_y)
    test_y = clf.predict(test_X)
    tokenizer.write_output('test',test_y)
    
    print('dev lables:')
    print(dev_labels)
    print("*" * 20)
    print('predicts:')
    print(dev_y)

  