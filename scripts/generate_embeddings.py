###################################
	
	# Ismael Bonneau

# Generates word2vec embeddings based on robust 2004 collection
# this script needs collection_tojson.py to be run

# parameters from the article a deep relevance matching model for ad-hoc retrieval https://arxiv.org/pdf/1711.08611.pdf
# 	vector size: 300
# 	window size: 10
#	min TF: 10 times
# 	# of negative samples: 10
#	subsampling threshold: 1e-4

# Text is stemmed using Krovetz stemmer

###################################


from time import time
import json
import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


class MySentences(object):
	def __init__(self):
		self.collections = ["FR94", "FT", "FBIS", "LATIMES"]
 
	def __iter__(self):
		for collection in self.collections:
			with open("/home/ismael/Documents/semantic_deep_neuralIR/data/robust2004"+collection+".json", "r") as f:
				coll = json.load(f)
				for doc_id in coll:
					yield coll[doc_id]['text'].split()
 
sentences = MySentences() # a memory-friendly iterator

import multiprocessing
from gensim.models import Word2Vec


cores = multiprocessing.cpu_count() # Count the number of cores in a computer

print("# of available cores: %d" % cores)

w2v_model = Word2Vec(min_count=10,
                     window=10,
                     size=300,
                     sample=1e-4, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=10,
                     workers=cores-1) # paramaters suggested in the article

# Building the vocab
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# Training the model
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

#w2v_model.init_sims(replace=True)

w2v_model.save('/home/ismael/Documents/semantic_deep_neuralIR/embeddings/model_1')
