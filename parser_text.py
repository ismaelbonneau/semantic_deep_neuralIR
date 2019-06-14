from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces, split_alphanum

CUSTOM_FILTERS = [strip_tags, strip_punctuation, strip_numeric,
                 strip_multiple_whitespaces, split_alphanum]


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
    return " ".join(preprocess_string(head.strip(), [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces]))

