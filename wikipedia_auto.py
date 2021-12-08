import json
import math
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
        
        punctuation = [(",", " ,"), (".", " ."), ("'", "' "), (":", " :"), (";", " ;"), ("(", "( "), (")", " )")]

        for s in d["simple"]["content"].keys():
            if s.startswith(p[0]):
                simple_par.append(d["simple"]["content"][s])
        
        for s in d["normal"]["content"].keys():
            if s.startswith(p[1]):
                normal_par.append(d["normal"]["content"][s])
        
        normal_text = " ".join(simple_par)
        simple_text = " ".join(normal_par)
        for x in punctuation:
            normal_text = normal_text.replace(x[0], x[1])
        for x in punctuation:
            simple_text = simple_text.replace(x[0], x[1])

        simple.append(normal_text)
        normal.append(simple_text)

train_len = math.floor(len(simple)*0.8)

with open('wiki_simple_train.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(simple[:train_len]))

with open('wiki_simple_test.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(simple[train_len:]))

with open('wiki_normal_train.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(normal[:train_len]))

with open('wiki_normal_test.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(normal[train_len:]))