import numpy as np
import pickle
import ast
import random
import os

#random.seed(13) #ti as compris



with open("data/robust2004.txt", "r") as f:
    queries = ast.literal_eval(f.read())
queries = list(queries.keys())

random.shuffle(queries)
if "634" in queries:
	queries.remove("634")
if "672" in queries:
	queries.remove("672")

print(len(queries), " queries.")

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

folds = list(split(queries, 5))

pickle.dump(folds, open("folds.pkl", "wb"))


