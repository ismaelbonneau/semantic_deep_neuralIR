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


import numpy as np
import gensim
import random
import ast
import json
import os
import pickle
from bs4 import BeautifulSoup
from os import listdir,sep
from os.path import isfile, join
from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,strip_numeric, strip_tags, strip_punctuation, strip_short
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import codecs
from gensim.models.wrappers import FastText


import string
table = str.maketrans('', '', '!"#$%\'()*+,-./:;<=>?@[\\]^_`{|}~')

def custom_tokenizer(s):
    return [w.translate(table) for w in preprocess_string(s, [lambda x: x.lower(), strip_tags, lambda x: strip_short(x, 2), remove_stopwords])]


class Dataset:

	def __init__(self, intervals, model_wv, normalize=False):
		"""
		all_doc : dictionnaire de tout nos documents afin d'itérer dessus. 
		"""
		self.intervals = intervals
		self.normalize = normalize
		self.model_wv = model_wv

	def load_idf(self, idf_file):
		self.idf_values = pickle.load(open(idf_file, "rb"))

	def get_vocab(self):
		return self.model_wv.wv.vocab

	def get_query(self, key, affiche=True):
		if affiche:
			print("query: ",key," ",self.d_query[key])
		return self.d_query[key]

	def get_doc(self,key,affiche=True):
		if affiche:
			print("doc: ",key," ",self.docs[key])
		return self.docs[key]

	def get_relevance(self,q_id,affiche=True):
		if affiche: 
			print("query: ",q_id," ",self.paires[q_id]['relevant'])
		return self.paires[q_id]['relevant']

	def get_idf_vec(self, query):
		"""
		"""
		vec = np.ones(self.max_length_query)
		for i, queryterm in enumerate(query):
			if queryterm.lower() in self.idf_values: #forcer la mise en majuscules on sait jamais
				vec[i] = self.idf_values[queryterm.lower()]
		return vec


	def load_all_query(self, file_query="data/robust2004.txt"):
		"""
			On recupère toutes les querys qui sont ensuite sauvegardées dans un dictionnaire. 
		"""
		with open(file_query, "r") as f:
			self.d_query = ast.literal_eval(f.read())

		for k in self.d_query :
			self.d_query[k]= self.d_query[k][0] # On suppr les query langage naturel, et on garde que la query mot clé
		self.max_length_query =  np.max([len(self.d_query[q].split()) for q in self.d_query])
		print("query chargé\n")


	def load_all_docs(self):
		"""
			Charge tout les docs dans un dico. 
		"""
		self.docs = {}
		collections = ["FR94", "FT", "FBIS", "LATIMES"]
		for collection in collections:
			with open("data/robust2004"+collection+".json", "r") as f:
				self.docs.update(json.load(f))
		print("docs chargés\n")

	def load_relevance(self, file_rel="data/qrels.robust2004.txt"):
		"""
			Chargement du fichier des pertinences pour les requêtes. 
			Pour chaque paire query/doc on nous dit si pertinent ou non. 
		"""
		self.paires = {}
		with open(file_rel,"r") as f:
			for line in f :
				l = line.strip().split(' ')
				self.paires.setdefault(l[0],{})
				self.paires[l[0]].setdefault('relevant',[])
				self.paires[l[0]].setdefault('irrelevant',[])
				if l[-1]=='1':
					self.paires[l[0]]['relevant'].append(l[2])
				else:
					self.paires[l[0]]['irrelevant'].append(l[2])
		
		print("relevance chargé\n")

	def embedding_exist(self, term):
		if term in self.model_wv:
			return term
		elif term.upper() in self.model_wv:
			return term.upper()
		else:
			return False


	def hist(self, query, document):
		"""
		query: matrice (nbtermequery x vector_size)
		document: matrice (nbtermedocument x vector_size)
		"""
		return np.apply_along_axis(lambda x: np.histogram(x, self.intervals)[0], 1, np.dot(query, document.T))


	def prepare_data_forNN(self, test_size=0.2):
		"""
		"""

		#spliter les requêtes en train/test
		lol = [q for q in self.d_query.keys() if q in self.paires]
		random.shuffle(lol)
		test_keys = lol[:int(test_size * len(lol))]
		train_keys = lol[int(test_size * len(lol)):]
		print("%d requetes en train, %d en test" % (len(train_keys), len(test_keys)))
		
		#pour chaque requête on va générer autant de paires relevant que irrelevant
		#pour nos besoins on va alterner paires positives et paires négatives
		train_hist = [] # les histogrammes d'interraction
		test_hist = []
		train_idf = [] #les vecteurs d'idf
		test_idf = []
		
		for id_requete in train_keys:
			#recuperer les mots dont on connait les embeddings dans la query
			query_embeddings = np.zeros((self.max_length_query, 300))
			i = 0
			for word in custom_tokenizer(self.d_query[id_requete]):
				if word in self.model_wv:
					query_embeddings.append(self.model_wv[word])
			query_embeddings = np.array(query_embeddings)

			idf_vec = self.get_idf_vec(q)
			for pos, neg in zip(self.paires[id_requete]["relevant"], self.paires[id_requete]["irrelevant"]):
				#lire le doc, la requete et creer l'histogramme d'interraction
				pos_embeddings = []
				for word in custom_tokenizer(self.docs[pos]['text']):
					if word in self.model_wv:
						pos_embeddings.append(self.model_wv[word])
				pos_embeddings = np.array(pos_embeddings)

				train_hist.append(self.hist(query_embeddings, pos_embeddings)) #append le doc positif
				train_idf.append(idf_vec) #append le vecteur idf de la requête
				
				neg_embeddings = []
				for word in custom_tokenizer(self.docs[neg]['text']):
					if word in self.model_wv:
						neg_embeddings.append(self.model_wv[word])
				neg_embeddings = np.array(neg_embeddings)

				train_hist.append(self.hist(query_embeddings, neg_embeddings)) #append le doc négatif
				train_idf.append(idf_vec) #append le vecteur idf de la requête

		train_labels = np.zeros(len(train_hist))
		train_labels[::2] = 1 # label de pertinence 
		print("train data completed")
		
		"""
		for id_requete in test_keys:
			#recuperer les mots dont on connait les embeddings dans la query
			query_embeddings = np.zeros((self.max_length_query, 300))
			i = 0
			for word in custom_tokenizer(self.d_query[id_requete]):
				if word in self.model_wv:
					query_embeddings[i] = self.model_wv[word]
					i+=1

			idf_vec = self.get_idf_vec(q)
			for pos, neg in zip(self.paires[id_requete]["relevant"], self.paires[id_requete]["irrelevant"]):
				#lire le doc, la requete et creer l'histogramme d'interraction
				pos_embeddings = []
				for word in custom_tokenizer(self.docs[pos]['text']):
					if word in self.model_wv:
						pos_embeddings.append(self.model_wv[word])
				pos_embeddings = np.array(pos_embeddings)

				test_hist.append(self.hist(query_embeddings, pos_embeddings)) #append le doc positif
				test_idf.append(idf_vec) #append le vecteur idf de la requête
				
				neg_embeddings = []
				for word in custom_tokenizer(self.docs[neg]['text']):
					if word in self.model_wv:
						neg_embeddings.append(self.model_wv[word])
				neg_embeddings = np.array(neg_embeddings)

				test_hist.append(self.hist(query_embeddings, neg_embeddings)) #append le doc négatif
				test_idf.append(idf_vec) #append le vecteur idf de la requête

		test_labels = np.zeros(len(train_hist))
		test_labels[::2] = 1
		print("test data completed")
		"""

		return (train_hist, train_idf, train_labels), test_keys
		
		#éventuellement sauvegarder tout ça sur le disque comme ça c fait une bonne fois pour toutes...


# La Ducance