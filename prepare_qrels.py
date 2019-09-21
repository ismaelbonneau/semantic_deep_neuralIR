import pickle
from functools import reduce

folds = pickle.load(open("folds.pkl", 'rb'))

qrel_file_path = "/home/ismael/Documents/semantic_deep_neuralIR/data/qrels.robust2004.txt"


paires = {}
with open(qrel_file_path,"r") as f:
	for line in f :
		l = line.strip().split(' ')
		paires.setdefault(l[0],{})
		paires[l[0]].setdefault('relevant',[])
		paires[l[0]].setdefault('irrelevant',[])
		if l[-1]=='1':
			paires[l[0]]['relevant'].append(l[2])
		else:
			paires[l[0]]['irrelevant'].append(l[2])
		
	print("relevance chargé")

def get_train(i, folds):
    train_fold = folds.copy()
    test_fold = train_fold.pop(i)
    train_fold = reduce(lambda x,y :x+y, train_fold)
    return train_fold

for i, fold in enumerate(folds):
	results = []
	for req in fold:
		results.extend(["{} 0 {} 1".format(req, id_doc) for id_doc in paires[req]['relevant']])
		results.extend(["{} 0 {} 0".format(req, id_doc) for id_doc in paires[req]['irrelevant']])

	with open("data/qrels_test_fold{}.txt".format(i), "w") as f:
		f.write("\n".join(results))


for i in range(len(folds)):
	fold = get_train(i, folds)
	results = []
	for req in fold:
		results.extend(["{} 0 {} 1".format(req, id_doc) for id_doc in paires[req]['relevant']])
		results.extend(["{} 0 {} 0".format(req, id_doc) for id_doc in paires[req]['irrelevant']])

	with open("data/qrels_train_fold{}.txt".format(i), "w") as f:
		f.write("\n".join(results))