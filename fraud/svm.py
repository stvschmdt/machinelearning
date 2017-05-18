from sklearn import svm
from model import Model

class SVM(Model):
    def __init__(self, params={}, debug=0):
        Model.__init__(self, params, debug)

    def svm_base(self, xtrain, ytrain, xtest, ytest):
        clf = svm.SVC()
        clf.fit(xtrain, ytrain)
        # test
        predictions = [int(a) for a in clf.predict(xtest)]
        num_correct = sum(int(a == y) for a, y in zip(predictions, ytest))
        print "Baseline classifier using an SVM."
        print "%s of %s values correct." % (num_correct, len(ytest))

if __name__ == "__main__":
    s = SVM()
    
