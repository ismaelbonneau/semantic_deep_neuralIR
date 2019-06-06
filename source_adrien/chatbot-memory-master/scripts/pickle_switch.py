import sys
import os
from os import path

libpath = path.normpath(path.join(path.dirname(path.realpath(__file__)), os.pardir, "src"))
sys.path.append(libpath)

import pickle as pkl
import torch

import data
from datasets import Quora, Robust2004

sys.modules["dataset"] = data

quora_dc = Quora.dataclasses()
quora_torch = Quora.torch_dataset()
rb_dc = Robust2004.dataclasses()
rb_torch = Robust2004.torch_dataset()

del sys.modules["dataset"]

with open(Quora.dataclasses_path, "wb") as f:
    pkl.dump(quora_dc, f)


with open(Robust2004.dataclasses_path, "wb") as f:
    pkl.dump(rb_dc, f)

torch.save(quora_torch, Quora.torch_path)
torch.save(rb_torch, Robust2004.torch_path)