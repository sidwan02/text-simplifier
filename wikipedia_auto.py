import json
import math
import matplotlib.pyplot as plt
f = json.load(open('wiki-auto-part-1-data.json'))
articles = [v for k, v in f.items()]
simple = []
normal = []

longest_simple = 0
longest_normal = 0

normal_cutoff = 400
simple_cutoff = 200

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
        
        normal_text = " ".join(normal_par)
        simple_text = " ".join(simple_par)
        for x in punctuation:
            normal_text = normal_text.replace(x[0], x[1])
        for x in punctuation:
            simple_text = simple_text.replace(x[0], x[1])

        if len(normal_text.split()) > normal_cutoff or len(simple_text.split()) > simple_cutoff:
            continue

        longest_simple = max(longest_simple, len(simple_text.split()))
        longest_normal = max(longest_normal, len(normal_text.split()))

        simple.append(simple_text)
        normal.append(normal_text)

train_len = math.floor(len(simple)*0.8)
print("Longest simple line: " + str(longest_simple))
print("Longest normal line: " + str(longest_normal))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].hist([len(x.split()) for x in simple], bins=40)
axs[1].hist([len(x.split()) for x in normal], bins=40)

plt.show()

with open('wiki_simple_train.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(simple[:train_len]))

with open('wiki_simple_test.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(simple[train_len:]))

with open('wiki_normal_train.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(normal[:train_len]))

with open('wiki_normal_test.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(normal[train_len:]))