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
    m = x.max()
    y = np.exp(x - m)
    return y / np.sum(y)

def sgd(w, data, niter):
    nlabels, dim = w.shape
    for iter in range(niter):
        train_loss = 0.0
        for yi, xi in train_data:
            # We compute the prediction of model and loss
            prediction = softmax(np.dot(w, xi))
            train_loss += math.log(prediction[yi])
            # We compute the gradient w.r.t. to w
            target = np.zeros(nlabels)
            target[yi] = 1.0
            error = prediction - target
            gradient = error.reshape((nlabels, 1)) * xi.reshape((1, dim))
            # We apply the gradient step
            w = w - 0.5 * gradient
        print("Iter: %02d    Loss: %.4f" % (iter, train_loss / len(data)))
    return w

def predict(w, x):
    return np.argmax(softmax(np.dot(w, x)))

def compute_accuracy(w, valid_data):
    accuracy = 0.0
    for yi, xi in valid_data:
        yp = predict(w, xi)
        if yp == yi:
            accuracy += 1.0
    return accuracy / len(valid_data)

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

#for niter in range(5):
#    train_loss = 0
#    for yi, xi in train_data:
#        p = softmax(np.dot(w, xi))
#        e = np.zeros(len(label_dict))
#        e[yi] = 1.0
#        e = e - p
#        w = w + 0.1 * e.reshape((len(label_dict), 1)) * xi.reshape((1, len(word_dict)))
#        train_loss += math.log(p[yi])
#        
#    print(train_loss / len(train_data))

#acc = 0.
#for yi, xi in valid_data:
#    yp = np.argmax(softmax(np.dot(w, xi)))
#    if yp == yi:
#        acc += 1.
#print acc / len(valid_data)
