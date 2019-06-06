import os
import re
import os.path as path

import elasticsearch as es
import elasticsearch.helpers as helpers
from bs4 import BeautifulSoup

from parser_trec import clean


index_name = "robust2004-titles"
b = 0.5
k1 = 1
settings = {"settings": {
                "number_of_shards": 1,
                "index": {
                    "similarity": {
                        "default": {
                            "type":"BM25",
                            "b": b,
                            "k1": k1}
                        }
                    },},
                "mappings": {         
                    "trec": {
                        "properties": {
                            "title": {
                                "type": "text",
                                "index": "false"
                            },
                            "text": {
                                "type": "text",
                                "analyzer": "english"
                                }
                            }
                        }
                    }
                }
            

def bulking(docnos, texts, heads, index=index_name):
    bulk_data = []
    for docno, txt, head in zip(docnos, texts, heads):
        #if len(docno) == 0 or len(txt) == 0:
        #    continue
        
        txt = clean(txt)
        data_dict = {
                '_index': index_name,
                '_type': 'trec',
                '_id': docno,
                '_source': {
                    "text": txt,
                    "title": head
                }
            }
        bulk_data.append(data_dict)
    return bulk_data

def parser_title(doc):
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
    return head


def parse_all_generator():
    rootDataset= path.expandvars("$DATASET_HOME/as_projet/TREC-ADHOC")
    collections=["FR94", "FT","FB", "LA" ]
    nb_fails = 0
    for root, dirnames, filenames in os.walk(rootDataset):
        for filename in filenames:
            for s in collections:
                if s in filename and 'DTD' not in filename and 'README' not in filename:
                    path_name = os.path.join(root, filename)

                    with open(path_name, 'r', encoding="utf-8", errors="ignore") as f:
                        file_content = f.read()
                    
                    # file_content = file_content.replace('\n', ' ').replace('\r', ' ')
                    # contents = file_content.split("<DOC>")
                    # contents = [c.strip() for c in contents[1:]]
                    
                    # docnos = [re.findall('<DOCNO>(.+?)</DOCNO>', c) for c in contents]
                    # docnos = map(lambda s: s[0].strip(), docnos)

                    # texts = [re.findall('<TEXT>(.+?)</TEXT>', c) for c in contents]
                    # texts = map(lambda s: clean(s), texts)
                    
                    # bulk_data = bulking(docnos, texts)


                    soup = BeautifulSoup(file_content, "html.parser")
                    docs = soup.find_all("doc")
                    
                    docnos, texts, heads = [], [], []
                    for doc in docs:
                        text = doc.text
                        docno = doc.docno.get_text()

                        head = parser_title(doc)
                        if len(head) == 0:
                            nb_fails += 1
                        
                        heads.append(head)
                        docnos.append(docno)
                        texts.append(text)

                    bulk_data = bulking(docnos, texts, heads)

                    yield bulk_data
    print("Some", nb_fails, "documents didn't have titles")


# With ES server started
engine = es.Elasticsearch()
assert engine.ping()

engine.indices.create(index_name, settings)

for bulk_data in parse_all_generator():
    helpers.bulk(engine, bulk_data)
