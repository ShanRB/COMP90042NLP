"""
No need to run this file.
This file will be called by get_features.py for preprocessing

Author: Rongbing Shan
"""
import os,json, re, inflect, unicodedata
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer

from textblob import TextBlob
CONTRACTIONS = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}


def get_sentiment_score(text):
    return TextBlob(text).sentiment

def expand_contractions(text):
    """Expand contractions in full form"""
    for word in text.split():
        if word in CONTRACTIONS:
            text = text.replace(word,CONTRACTIONS[word])
    return text

def remove_single_character(tokens):
    return [token for token in tokens if len(token) > 1]

def remove_urls(text):
    re_exp = '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
    has_url = re.search(re_exp, text)
    if has_url:
        urls = re.findall(re_exp, text)
        new_text = re.sub(re_exp,'',text)
        return len(urls),new_text
    else:
        return 0,text

def remove_non_ascii(tokens):
    new_words = []
    for word in tokens:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def remove_punctuation(tokens):
    new_words = []
    for word in tokens:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(tokens):
    p = inflect.engine()
    new_words = []
    for word in tokens:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(tokens):
    stopword = set(stopwords.words('english'))
    new_words = []
    for word in tokens:
        if word not in stopword:
            new_words.append(word)
    return new_words

def stemming(tokens):
    stemmer = LancasterStemmer()
    stems = []
    for word in tokens:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in tokens:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def tokenize(document):
    tokens = []
    document = expand_contractions(document.replace("‘","'").lower())
    num_urls, document = remove_urls(document)
    sentences = sent_tokenize(document)
    for sentence in sentences:
        temp_tokens = word_tokenize(sentence)
        tokens += temp_tokens
    return num_urls,tokens

def normalize(tokens):
    tokens = remove_non_ascii(tokens)
    tokens = remove_punctuation(tokens)
    #tokens = replace_numbers(tokens)
    tokens = remove_stopwords(tokens)
    return tokens

def pre_process(jsonfile):
    urls = []
    stem = []
    for key in jsonfile:
        doc = jsonfile[key]["text"]
        num_urls,docs = tokenize(doc)
        tokens = normalize(docs)
        tokens = lemmatize(tokens)
        tokens = stemming(tokens)
        stem.append(tokens)
        urls.append([num_urls])
    return urls,stem

def write_output(type,y):
    result = {}
    for i in range(0,len(y)):
        key = f'{type}-{i}'
        if y[i] == 1:
            label = 1
        else:
            label = 0
        result[key] = {"label":label}
    outputfilename = f'{type}-output.json'
    print('write result to ', outputfilename)
    with open(outputfilename,'w') as output:
        json.dump(result,output)
    if type == 'test':
        os.system("zip rs_colab.zip test-output.json")

def getlabels(Jsonfile):
    labels = []
    for key in Jsonfile:
        labels.append([Jsonfile[key]['label']])
    return labels

if __name__ == '__main__':
    devfilename = 'dev.json'
    devfile = open(devfilename)
    devJson = json.load(devfile)
    devfile.close()
    print(getlabels(devJson))

    