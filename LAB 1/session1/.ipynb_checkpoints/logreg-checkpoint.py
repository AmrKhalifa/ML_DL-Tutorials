import io, sys, math
import numpy as np
from collections import defaultdict

def build_dict(filename, threshold=1):
    fin = io.open(filename, 'r', encoding='utf-8')
    word_dict, label_dict = {}, {}
    counts = defaultdict(lambda: 0)
    for line in fin:
        tokens = line.split()
        label = tokens[0]

        if not label in label_dict:
            label_dict[label] = len(label_dict)

        for w in tokens[1:]:
            counts[w] += 1
            
    for k, v in counts.iteritems():
        if v > threshold:
            word_dict[k] = len(word_dict)
    return word_dict, label_dict

def load_data(filename, word_dict, label_dict):
    fin = io.open(filename, 'r', encoding='utf-8')
    data = []
    dim = len(word_dict)
    for line in fin:
        tokens = line.split()
        label = tokens[0]

        yi = label_dict[label]
        xi = np.zeros(dim)
        for word in tokens[1:]:
            if word in word_dict:
                wid = word_dict[word]
                xi[wid] += 1.0
        data.append((yi, xi))
    return data

def softmax(x):
    ## FILL CODE
    return None

def sgd(w, data, niter):
    nlabels, dim = w.shape
    for iter in range(niter):
        ## FILL CODE
    return w

def predict(w, x):
    ## FILL CODE
    return None

def compute_accuracy(w, valid_data):
    ## FILL CODE
    return 0.0

print("")
print("** Logistic Regression **")
print("")

word_dict, label_dict = build_dict(sys.argv[1])
train_data = load_data(sys.argv[1], word_dict, label_dict)
valid_data = load_data(sys.argv[2], word_dict, label_dict)

nlabels = len(label_dict)
dim = len(word_dict)
w = np.zeros([nlabels, dim])
w = sgd(w, train_data, 5)
print("")
print("Validation accuracy: %.3f" % compute_accuracy(w, valid_data))
print("")
