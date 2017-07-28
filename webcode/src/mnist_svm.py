"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

import mnist_loader 
from sklearn import svm
from sklearn.externals import joblib

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    #print training_data[0][0].shape, training_data[0][0].shape
    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
    joblib.dump(clf, '/home/ubuntu/machinelearning/src/nicefolk/simple_svm.pkl')

if __name__ == "__main__":
    svm_baseline()
    
