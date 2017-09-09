#author: steve schmidt
#simple sgd for mnist data
#functions for simple classification
#as well as extensions for new tensorflow sgd

import time
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# timing wrapper
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print('timefn:' + fn.func_name + ' ' + str(t2-t1) + ' sec')
        return result
    return measure_time

#simple sgd
def train_sgd(X, y):
    sgd_clf = SGDClassifier()
    sgd_clf.fit(X, y)
    return sgd_clf

def predict_sgd(X, clf):
    clf.predict(X)
    return clf

def x_validate(X, y,clf):
    x_validate = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    x_predict = cross_val_predict(clf, X, y, cv=3)
    return x_validate, x_predict

def confusion_mat(y_train, y_pred):
    matrix = confusion_matrix(y_train, y_pred)
    return matrix

def precision_recall_f1(y, y_pred, ave='macro'):
    prec = precision_score(y, y_pred, average=ave)
    rec = recall_score(y, y_pred, average=ave)
    f = f1_score(y, y_pred, average=ave)
    return prec, rec, f

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']
    #test_img = X[36000]
    #test_img = test_img.reshape(28,28)
    #plt.imshow(test_img, interpolation='nearest')
    #plt.axis('off')
    #plt.show()
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_idx = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]
    train_clf = train_sgd(X_train, y_train)
    predict_clf = predict_sgd(X_test, train_clf)
    validate = x_validate(X_test, y_test, predict_clf)
    con_matrix = confusion_mat(y_test, validate[1])
    prec, rec, f1 = precision_recall_f1(y_test, validate[1])
    print validate[0]
    print con_matrix
    print prec
    print rec
    print f1
