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
        label_counts[label] += 1.0
        label_total += 1.0
        for w in sentence:
            word_counts[label][w] += 1.0
            word_total[label] += 1.0

    return {'label_counts': label_counts, 
            'word_counts': word_counts, 
            'label_total': label_total, 
            'word_total': word_total}

def predict(sentence, mu, label_counts, word_counts, label_total, word_total):
    best_label = None
    best_score = float('-inf')

    for label in word_counts.keys():
        score = 0.0
        voc_size = len(word_counts[label])
        for w in sentence:
            wc = word_counts[label][w] + mu
            tc = word_total[label] + mu * voc_size
            score += math.log(wc / tc)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label

def compute_accuracy(valid_data, mu, counts):
    accuracy = 0.0
    for label, sentence in valid_data:
        prediction = predict(sentence, mu, **counts)
        if prediction == label:
            accuracy += 1.0
    return accuracy / len(valid_data)

print("")
print("** Naive Bayes **")
print("")

mu = 1.
train_data = load_data(sys.argv[1])
valid_data = load_data(sys.argv[2])
counts = count_words(train_data)

print("Validation accuracy: %.3f" % compute_accuracy(valid_data, mu, counts))
print("")
