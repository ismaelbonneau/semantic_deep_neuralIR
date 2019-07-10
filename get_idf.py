# J'essaie le chargement des idf avec sklearn mais tu peux essayer avec elasticsearch'
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import json
import pickle

from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces, split_alphanum

from krovetzstemmer import Stemmer #stemmer pas mal pour la PR

ks = Stemmer()

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_multiple_whitespaces, strip_punctuation, remove_stopwords, lambda x: ks.stem(x)]

corpus = []
for jsn in ['robust2004FBIS.json', 'robust2004FR94.json', 'robust2004FT.json', 'robust2004LATIMES.json']:
    with open('data/'+jsn, 'r') as f: 
        collection = json.load(f)
        for doc in collection:
            corpus.append(collection[doc]['text'])
        print("collection %s done." % jsn)

vectorizer = TfidfVectorizer(
                        use_idf=True,
                        smooth_idf=False, 
                        sublinear_tf=False,
                        binary=False,
                        min_df=1, max_df=1.0, max_features=None,
                        ngram_range=(1,1))

X = vectorizer.fit(corpus) #osef du transform
idf = vectorizer.idf_
idf = dict(zip(vectorizer.get_feature_names(), idf)) #notre dictionnaire terme -> idf

pickle.dump(idf, open("idf_robust2004.pkl", "wb"))