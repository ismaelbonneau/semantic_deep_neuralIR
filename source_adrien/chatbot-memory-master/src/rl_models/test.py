import gc
print(len(gc.get_objects()))
import torch
print(len(gc.get_objects()))
g = gc.get_objects()
print(type(g))
for i in g:
    print(i)
