import io, sys, math, re
from collections import defaultdict

def load_data(filename):
    fin = io.open(filename, 'r', encoding='utf-8')
    data = []
    for line in fin:
        tokens = line.split()
        data.append((tokens[0], tokens[1:]))
    return data

def count_words(data):
    label_total = 0
    word_total = defaultdict(lambda: 0)
    label_counts = defaultdict(lambda: 0)
    word_counts = defaultdict(lambda: defaultdict(lambda: 0.0))

    for example in data:
        label, sentence = example
        ## FILL CODE

    return {'label_counts': label_counts, 
            'word_counts': word_counts, 
            'label_total': label_total, 
            'word_total': word_total}

def predict(sentence, mu, label_counts, word_counts, label_total, word_total):
    best_label = None
    best_score = float('-inf')

    for label in word_counts.keys():
        score = 0.0
        ## FILL CODE

    return best_label

def compute_accuracy(valid_data, mu, counts):
    accuracy = 0.0
    for label, sentence in valid_data:
         ## FILL CODE
    return 0.0

print("")
print("** Naive Bayes **")
print("")

mu = 1.0
train_data = load_data(sys.argv[1])
valid_data = load_data(sys.argv[2])
counts = count_words(train_data)

print("Validation accuracy: %.3f" % compute_accuracy(valid_data, mu, counts))
print("")
