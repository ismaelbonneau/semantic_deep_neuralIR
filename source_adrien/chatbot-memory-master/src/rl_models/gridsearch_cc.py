import subprocess
import threading
import functools
import itertools

from itertools import chain

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()


def mul(it):
    res = 1
    for i in it:
        res *= i
    return res

def run_exp(parameter_yielder):
    for parameters in parameter_yielder:
        hsize, nlayer, batch, smt, ent, rl, epochs = parameters
        args = f"-d 2 -smt {smt} -rl {rl} -ent {ent} -n {epochs} -hs {hsize} -l {nlayer} -b {batch}".split(" ")
        print(threading.current_thread().name, "is running", " ".join(args))
        out_file = f"../../res/cc-{smt}-{rl}-{ent}_{batch}-{nlayer}-{hsize}.stdout"
        with open(out_file, "w") as f:
            subprocess.run(["python3",  "wafa.py", "c", "c", *args], stdout=f, stderr=subprocess.STDOUT)


def exp_parameters(*args):
    print(mul(map(len, args)), " combination to try")
    for comb in itertools.product(*args):
        yield comb

print("Grid Searching !")
hsizes = [80, 96, 112]
nlayers = [1]
batches = [1, 2, 4]
smt_lambda = [1]
ent_lambda = [0]
rl_lambda = [0]
epochs = [150]

print("Creating iterators")
first_pool = exp_parameters(hsizes, nlayers, batches, smt_lambda, ent_lambda, rl_lambda, epochs)

smt_lambda = [0]
rl_lambda = [0.01, 0.05, 0.1, 0.5, 1.0]
ent_lambda = [0.1, 0.5, 1.0, 1.5, 2.0]
epochs = [25]

second_pool = exp_parameters(hsizes, nlayers, batches, smt_lambda, ent_lambda, rl_lambda, epochs)
exp_param = LockedIterator(chain(first_pool, second_pool))
exp_runner = functools.partial(run_exp, exp_param)

threads = [
    threading.Thread(target=exp_runner)
    for i in range(15)
]

print("Starting Threads")
for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
print("Done")
