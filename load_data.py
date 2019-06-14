##########################################################
#
# Fichier contenant les fonctions de chargement des documents d'une collection
#
# Ismael Bonneau
##########################################################

import os
from bs4 import BeautifulSoup

from parser_text import get_docno, get_text, get_title


def bulking(docs, index_name):
	bulk_data = []
	for doc in docs:
		data_dict = {
				'_index': index_name,
				'_id': get_docno(doc),
				'_source': {
					"text": get_text(doc),
					"title": get_title(doc)
				}
			}
		bulk_data.append(data_dict)
	return bulk_data

def collect_all_data(datasetpath, collections, index_name='robust-2004'):

	listedocs = []
	for collection in collections:
		for root, dirs, files in os.walk(datasetpath+os.sep+collection, topdown=True):
			for name in files:
				with open(os.path.join(root, name), "r") as f:
					filecontent = f.read()
					soup = BeautifulSoup(filecontent, "html.parser")
					docs = soup.find_all("doc")

					yield bulking(docs, index_name)


# La Ducance