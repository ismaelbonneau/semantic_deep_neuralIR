pathQueries = "/local/pouyet/datasets/04.testset"

with open(pathQueries, 'r') as f:
    lines = f.read().split("\n")

lines = list(filter(lambda s: len(s.strip()) > 0, lines))


queries = {}
i = 0

while not i == len(lines):
    line = lines[i]
    if "<num>" in line:
        idQ = line.split('Number: ', 1)[-1].strip()
    else:
        i += 1
        continue
    
    i += 1
    while not "<title>" in lines[i]:
        i += 1
    title = lines[i].replace("<title>", "").strip()
    if len(title) == 0:
        i += 1
        title = lines[i].strip()
    title = title.replace("/", " ").replace("-", " ").replace("(", "").replace(")", "").lower()

    i += 1
    while not "<desc>" in lines[i]:
        i += 1
    
    i += 1
    desc = []
    while not "<narr>" in lines[i]:
        desc.append(lines[i])
        i += 1
    
    desc = " ".join(desc)
    desc = desc.replace("\n", "").replace("/", " ").replace("-", " ").replace(")", "").replace("(", "").lower()
    desc = desc.replace("\"", "").replace(".", "").replace(":", "")
    
    queries[idQ] = (title, desc)

path_save = "/local/pouyet/datasets/as_projet/querries/robust2004.txt"

with open(path_save, "w") as f:
    f.write(str(queries))
