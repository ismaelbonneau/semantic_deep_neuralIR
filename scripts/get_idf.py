###############################################
#           Ismael Bonneau

# Using sklearn' TfidfVectorizer to compute the inverse document frequency of every term in robust 2004.

# This script needs clean JSON robust04 collections so it has to be executed AFTER collection_tojson.py

###############################################


from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import json
import pickle
from time import time

t = time()
corpus = []
for jsn in ['robust2004FBIS.json', 'robust2004FR94.json', 'robust2004FT.json', 'robust2004LATIMES.json']:
    with open('/home/ismael/Documents/semantic_deep_neuralIR/data/'+jsn, 'r') as f: 
        collection = json.load(f)
        for doc in collection:
            corpus.append(collection[doc]['text'])
        print("collection %s loaded." % jsn)

vectorizer = TfidfVectorizer(
                        use_idf=True,
                        smooth_idf=False, 
                        sublinear_tf=False,
                        binary=False,
                        ngram_range=(1,1))

_ = vectorizer.fit(corpus)
idf = vectorizer.idf_
idf = dict(zip(vectorizer.get_feature_names(), idf)) # our term -> idf dictionnary

pickle.dump(idf, open("/home/ismael/Documents/semantic_deep_neuralIR/idf_robust2004.pkl", "wb"))
print("took {} minutes. done.".format(round((time() - t) / 60, 2)))