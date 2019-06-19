import json
import os
from bs4 import BeautifulSoup
from parser_text import get_docno, get_text, get_title
import codecs

collections = ['FR94','FT']
for collection in collections:
	documents = {}
	for root, dirs, files in os.walk("data"+os.sep+"collection"+os.sep+collection, topdown=True):
		for name in files:
			with codecs.open(os.path.join(root, name), "r", encoding="utf-8", errors="ignore") as f:
				filecontent = f.read()
				soup = BeautifulSoup(filecontent, "html.parser")
				docs = soup.find_all("doc")

				for doc in docs:
					print(get_docno(doc))
					documents[get_docno(doc)] = {'title': get_title(doc), 'text': get_text(doc)}
	print("collection %s done (collecting)" % collection)
	save = json.dumps(documents)
	with open("data/robust2004"+collection+".json","w") as f:
		f.write(save)
	print("collection %s saved" % collection)

#'FR94','FT'