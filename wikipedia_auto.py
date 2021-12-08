import json
f = json.load(open('wiki-auto-part-1-data.json'))
articles = [v for k, v in f.items()]
simple = []
normal = []

for d in articles:
    alignments = d["paragraph_alignment"]
    if len(alignments) == 0:
        continue
    for p in alignments:
        simple_par = []
        normal_par = []
        
        for s in d["simple"]["content"].keys():
            if s.startswith(p[0]):
                simple_par.append(d["simple"]["content"][s])
        
        for s in d["normal"]["content"].keys():
            if s.startswith(p[1]):
                normal_par.append(d["normal"]["content"][s])
        
        simple.append(" ".join(simple_par))
        normal.append(" ".join(normal_par))

with open('wiki_simple.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(simple))

with open('wiki_normal.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(normal))