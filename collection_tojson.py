import json
import os
from bs4 import BeautifulSoup
from parser_text import get_docno, get_text, get_title
import codecs

from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,strip_numeric, strip_tags, strip_punctuation, strip_short, strip_multiple_whitespaces

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_multiple_whitespaces, strip_punctuation, remove_stopwords, lambda x: ks.stem(x)]

import string
table = str.maketrans('', '', '!"#$%\'()*+,-./:;<=>?@[\\]^_`{|}~')

# Krovetz stemmer est un stemmer moins "destructif" que le porter.
# Viewing morphology as an inference process: https://dl.acm.org/citation.cfm?id=160718
from krovetzstemmer import Stemmer #stemmer pas mal pour la PR
ks = Stemmer()


def lol(x):
    return x.replace("&hyph", "")

def custom_tokenizer(s):
    return " ".join([lol(ks.stem(w.translate(table))) for w in preprocess_string(s, [lambda x: x.lower(), strip_tags, strip_numeric, lambda x: strip_short(x, 1), remove_stopwords])])

collections = ['FR94','FT', 'LATIMES', 'FBIS']
for collection in collections:
	documents = {}
	for root, dirs, files in os.walk("data"+os.sep+"collection"+os.sep+collection, topdown=True):
		for name in files:
			with codecs.open(os.path.join(root, name), "r", encoding="utf-8", errors="ignore") as f:
				filecontent = f.read()
				soup = BeautifulSoup(filecontent, "html.parser")
				docs = soup.find_all("doc")

				for doc in docs:
					documents[get_docno(doc)] = {'title': get_title(doc), 'text': custom_tokenizer(get_text(doc))}
	print("collection %s done (collecting)" % collection)
	save = json.dumps(documents)
	with open("data/robust2004"+collection+".json","w") as f:
		f.write(save)
	print("collection %s saved" % collection)

#'FR94','FT'