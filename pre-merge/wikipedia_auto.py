import json
import math
import matplotlib.pyplot as plt
from collections import Counter
import re

# Import file, initialize the list of paragraphs and the articles
f = json.load(open('wiki-auto-part-1-data.json'))
articles = [v for k, v in f.items()]
simple = []
normal = []

# Initialize counters for longest paragraphs
longest_simple = 0
longest_normal = 0

# Define cutoffs for lengths for paragraphs
normal_cutoff = 400
simple_cutoff = 200

# Define unking character and vocab size
unk = "*UNK*"
non_unk_vocab_size = 5000

class UnkReplacer:
    """
	Class to unk a list of infrequent words out of a given string.
	"""
    def __init__(self, non_replace, unk_char):
        self.non_replace = set(non_replace)
        self.unk = unk_char

    def __call__(self, strings):
        filter_words_from_str = lambda y: " ".join(map(lambda x: x if x in self.non_replace else self.unk, y.split()))
        return "\n".join(map(filter_words_from_str, strings))

# Main loop
for d in articles:
    # Get paragraph alignments, skip if there are none
    alignments = d["paragraph_alignment"]
    if len(alignments) == 0:
        continue

    # Iterate through alignments, matching appropriate paragraphs
    for p in alignments:
        simple_par = []
        normal_par = []
        
        # Initialize list of punctuation to add spaces to
        punctuation = [(",", " ,"), (".", " ."), ("'", " ' "), (":", " :"), (";", " ;"), ("(", "( "), (")", " )")]

        # Filter through content of simple to accumulate the ones in the current simple paragraph
        for s in d["simple"]["content"].keys():
            if s.startswith(p[0]):
                simple_par.append(d["simple"]["content"][s])
        
        # Filter through content of normal to accumulate the ones in the current normal paragraph
        for s in d["normal"]["content"].keys():
            if s.startswith(p[1]):
                normal_par.append(d["normal"]["content"][s])
        
        # Combine sentences
        normal_text = " ".join(normal_par)
        simple_text = " ".join(simple_par)

        # Replace spaced punctuation
        for x in punctuation:
            normal_text = normal_text.replace(x[0], x[1])
        for x in punctuation:
            simple_text = simple_text.replace(x[0], x[1])

        # Filter out paragraphs that are too long
        if len(normal_text.split()) > normal_cutoff or len(simple_text.split()) > simple_cutoff:
            continue
        
        # Update counter of longest paragraphs
        longest_simple = max(longest_simple, len(simple_text.split()))
        longest_normal = max(longest_normal, len(normal_text.split()))

        # Append to accumulator of simple and normal text
        simple.append(simple_text)
        normal.append(normal_text)

# Get the length of the training set, print longest paragraphs
train_len = math.floor(len(simple)*0.8)
print("Longest simple paragraph: " + str(longest_simple))
print("Longest normal paragraph: " + str(longest_normal))

"""
# Plot histogram of paragraph lengths
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].hist([len(x.split()) for x in simple], bins=40)
axs[1].hist([len(x.split()) for x in normal], bins=40)

plt.show()
"""

# Create filterer for unking
all_text = " ".join(simple) + " " + " ".join(normal)
word_freqs = Counter(all_text.split())
most_freq = word_freqs.most_common(non_unk_vocab_size)
non_replace = [word for word, _ in most_freq]
replacer = UnkReplacer(non_replace, unk)

# Unk data and save it, splitting it according to train_len
with open('wiki_simple_train.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write(replacer(simple[:train_len]))

with open('wiki_simple_test.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write(replacer(simple[train_len:]))

with open('wiki_normal_train.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write(replacer(normal[:train_len]))

with open('wiki_normal_test.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write(replacer(normal[train_len:]))