# J'essaie le chargement des idf avec sklearn mais tu peux essayer avec elasticsearch'
from parser_text import get_text
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup


corpus = []
for collection in ["FR94", "FT", "FBIS", "LATIMES"]:
    for root, dirs, files in os.walk("data/collection"+os.sep+collection, topdown=True):
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                try:
                    filecontent = f.read()
                    soup = BeautifulSoup(filecontent, "html.parser")
                    docs = soup.find_all("doc")
                    for doc in docs:
                        corpus.append(get_text(doc))
                except:
                    continue

vectorizer = TfidfVectorizer(
                        use_idf=True,
                        smooth_idf=True, 
                        sublinear_tf=False,
                        binary=False,
                        min_df=1, max_df=1.0, max_features=None,
                        ngram_range=(1,1))

X = vectorizer.fit(corpus) #osef du transform
idf = vectorizer.idf_
idf = dict(zip(vectorizer.get_feature_names(), idf)) #notre dictionnaire terme -> idf

pickle.dump(idf, open("idf_robust2004.pkl", "wb"))