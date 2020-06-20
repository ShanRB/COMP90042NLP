"""
No need to run this file.
This file will be called by one-class-svm.py, randomforest.py and svm.py
to generate features used in training process.

Author: Rongbing Shan
"""
import tokenizer
import numpy as np
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import nltk
from nltk.parse import CoreNLPParser
from nltk.tree import Tree
from sentistrength import PySentiStr


#nltk.download('averaged_perceptron_tagger')

tag_sets = ['$','CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP',\
    'NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD',\
        'VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
causual_words = tokenizer.lemmatize(['because','so','since','therefore','hence','consequently','though','even',\
    'yet','as','accordingly','result','reason','make','how','why','lead','due','Owing'])
contrast_words = tokenizer.lemmatize(['although','than','besides','nevertheless','compare','converse','otherwise',\
    'differ','regardless','furthermore','though','however','unless','unlike','contrast',\
        'instead','while','yet'])
negation_words = ('no','not','never')

def get_features(json_file):
    all_tokens = []
    all_postags = []
    all_otherfeature = []
    for key in json_file:
        tokens = []
        other_feat = {}
        postag_dict = {key:0 for key in tag_sets}
        document = json_file[key]['text']

        # expand contractions to normal form
        document = tokenizer.expand_contractions(document.replace("‘","'").lower())
        num_urls, document = tokenizer.remove_urls(document)
        other_feat['url_count'] = num_urls
        
        # sentence segmentation
        sentences = sent_tokenize(document)
        for sentence in sentences:
            temp_tokens = word_tokenize(sentence)
            tokens += temp_tokens
        num_sentences = len(sentences)
        num_tokens = len(tokens)
        other_feat['avg_word_per_sent'] = num_tokens * 1.0/ num_sentences

        # remove single character
        tokens = tokenizer.remove_single_character(tokens)
        #sentiment analysis
        result = tokenizer.get_sentiment_score(document)
        other_feat['sentiment'] = result[0]
        other_feat['objectivity'] = result[1]
        neg_count = 0
        pos_count = 0
        min_senti = 0
        max_senti = 0
        for sentence in sentences:
            sentiscore = tokenizer.get_sentiment_score(sentence)
            if sentiscore[0] >= 0 :
                pos_count += 1
            else:
                pos_count += 1
            if sentiscore[0] < min_senti:
                min_senti = sentiscore[0]
            if sentiscore[0] > max_senti:
                max_senti = sentiscore[0]
        other_feat['pos_sentence_count'] = pos_count
        other_feat['neg_sentence_count'] = neg_count
        other_feat['min_senti_score'] = min_senti
        other_feat['max_senti_score'] = max_senti

        #lemmatization
        tokens = tokenizer.lemmatize(tokens)

        #sepcial words
        list_causual = [word for word in tokens if word in causual_words]
        list_contrast = [word for word in tokens if word in contrast_words]
        list_negation = [word for word in tokens if word in negation_words]
        other_feat['causual'] = len(list_causual)
        other_feat['contrast'] = len(list_contrast)
        other_feat['negation'] = len(list_negation)

        # stop words
        count_before = len(tokens)
        tokens = tokenizer.remove_stopwords(tokens)
        num_stopwords = count_before - len(tokens)
        other_feat['stopwords_pct'] = num_stopwords * 1.0 / num_tokens

        # remove non ascii and puncturation
        tokens = tokenizer.remove_non_ascii(tokens)
        count_before = len(tokens)
        tokens = tokenizer.remove_punctuation(tokens)
        num_punc = count_before - len(tokens)
        other_feat['punctuations_pct'] = num_punc * 1.0 / num_tokens

        #lemmatization
        tokens = tokenizer.lemmatize(tokens)

        # lexical diversity
        other_feat['lexical_diversity'] = len(set(tokens))*1.0/num_tokens

        # count POS tags
        postags = pos_tag(tokens)
        for pos in postags:
            if pos[1] in postag_dict:
                postag_dict[pos[1]] += 1

        """
        # parse tree
        parser = CoreNLPParser(url='http://localhost:9000')
        parsetree = list(parser.parse(tokens))[0]
        other_feat['tree_height'] = parsetree.height()
        for s in parsetree.subtrees():
            print(s)
        """
        

        all_tokens.append(tokens)
        all_postags.append(list(postag_dict.values()))
        all_otherfeature.append(list(other_feat.values()))

    return all_tokens,all_postags,all_otherfeature



if __name__ == '__main__':
    trainfilename = 'train.json'
    devfilename = 'dev.json'
    testfilename = 'test-unlabelled.json'

    trainfile = open(trainfilename)
    trainJson = json.load(trainfile)
    trainfile.close()

    train_tokens,train_tags,train_otherfeats = get_features(trainJson)
