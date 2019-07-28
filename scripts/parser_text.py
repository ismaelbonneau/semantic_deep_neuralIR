from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces, split_alphanum

# pip install -i https://test.pypi.org/simple/ --only-binary=krovetz krovetz

import krovetz #stemmer pas mal

# Krovetz stemmer est un stemmer moins "destructif" que le porter.
# Viewing morphology as an inference process: https://dl.acm.org/citation.cfm?id=160718

ks = krovetz.PyKrovetzStemmer()

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_multiple_whitespaces, strip_punctuation, remove_stopwords, lambda x: ks.stem(x)]


def get_docno(doc):
    return doc.docno.get_text().strip()

def get_text(doc):
    return " ".join(preprocess_string(doc.text, CUSTOM_FILTERS))

def get_title(doc):
    head = ""
    try:
        head = doc.headline.get_text()    
    except:
        pass

    try:
        head = doc.h3.get_text()
    except:
        pass

    try:
        head = doc.doctitle.get_text()
    except:
        pass
    return " ".join(preprocess_string(head.strip(), CUSTOM_FILTERS))