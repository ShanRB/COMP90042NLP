from nltk.corpus import wordnet as wn
syn = wn.synsets('compare')[0] 
  
print ("Synset name :  ", syn.name()) 
  
# Defining the word 
print ("\nSynset meaning : ", syn.definition()) 
  
# list of phrases that use the word in context 
print ("\nSynset example : ", syn.examples()) 

print ("\nSynset specific term :  ",  
       syn.hypernyms()[0].hyponyms()) 