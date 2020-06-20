"""
To generate result for One Class SVM
This file will be runned using one class svm, the result file wil be saved to
dev-output.json and test-output.json

Input   : train.json, dev.json, test-unlabelled.json
Output  : dev-output.json, test-output.json
"""
import json
import tokenizer
import numpy as np
from get_features import get_features
from scipy.sparse import hstack,csr_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

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
    
    # acquire features for training
    train_tokens,train_tags,train_otherfeats = get_features(trainJson)

    #vec = TfidfVectorizer(tokenizer=dummy,preprocessor=dummy)
    vec = CountVectorizer(tokenizer=dummy,preprocessor=dummy)
    train_vec = vec.fit_transform(train_tokens).toarray()
    train_tags = csr_matrix(train_tags).toarray()
    train_otherfeats = csr_matrix(train_otherfeats).toarray()
    train_X = np.concatenate([train_otherfeats,train_tags,train_vec],axis=1)
    #train_X = train_vec
    print('shape of train X: ', train_X.shape)

    # acquire dev features
    dev_tokens,dev_tags,dev_otherfeats = get_features(devJson)
    dev_labels = tokenizer.getlabels(devJson)
    dev_vec = vec.transform(dev_tokens).toarray()
    dev_tags = csr_matrix(dev_tags).toarray()
    dev_otherfeats = csr_matrix(dev_otherfeats).toarray()
    dev_X = np.concatenate([dev_otherfeats,dev_tags,dev_vec],axis=1)
    #dev_X = dev_vec
    print('shape of dev X: ', dev_X.shape)
    # acquire test features
    test_tokens,test_tags,test_otherfeats = get_features(testJson)
    test_vec = vec.transform(test_tokens).toarray()
    test_tags = csr_matrix(test_tags).toarray()
    test_otherfeats = csr_matrix(test_otherfeats).toarray()
    test_X = np.concatenate([test_otherfeats,test_tags,test_vec],axis=1)
    
    
    # one class svm
    all_f = []
    all_p = []
    all_r = []
    nus = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for nu in nus:
        ocsvm = OneClassSVM(gamma='scale', nu=nu)
        ocsvm.fit(train_X)
        dev_y = ocsvm.predict(dev_X)
        for n,y in enumerate(dev_y):
            if y == -1:
                dev_y[n] = 0
        p, r, f, _ = precision_recall_fscore_support(\
             dev_labels, dev_y, pos_label=1, average="binary")
        all_f.append(f)
        all_p.append(p)
        all_r.append(r)

    ocsvm_best = max(all_f)
    best_nu = nus[all_f.index(ocsvm_best)]
    print(all_f)
    print(all_p)
    print(all_r)
    print(ocsvm_best, ' - ', best_nu)

    ocsvm = OneClassSVM(gamma='scale', nu=best_nu)
    ocsvm.fit(train_X)
    dev_y = ocsvm.predict(dev_X)
    tokenizer.write_output('dev',dev_y)
    test_y = ocsvm.predict(test_X)
    tokenizer.write_output('test',test_y)

    print('dev lables:')
    print(dev_labels)
    print("*" * 20)
    print('predicts:')
    print(dev_y)