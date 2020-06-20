"""
collect data f rom event registry
"""
from eventregistry import *
import json

er = EventRegistry(apiKey = "ca169245-d758-448c-ae85-b8820f015134")
q = QueryArticlesIter(
    keywords = "climate change",
    startSourceRankPercentile=0,
    endSourceRankPercentile=20)
num = 0
data = {}
for article in q.execQuery(er, sortBy = "rel", maxItems = 1500):
    content ={}
    key = f'train-{num}'
    content['text'] = article['body']
    content['label'] = 0
    data[key] = content
    #print(article['body'])
    num += 1
                                        
print(num)

filename = "train_eventregister.json"
with open(filename,'w') as output:
    json.dump(data,output)
