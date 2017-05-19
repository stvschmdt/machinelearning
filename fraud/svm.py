from sklearn import svm

from model import Model
from processing import Processor
from logger import Logging

class SVM(Model):
    def __init__(self, params={}, data=None, debug=0):
        Model.__init__(self, params, debug)
        self.clf = svm.SVC()
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None

    def svm_run(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.svm_fit()
        self.svm_predict()
        self.svm_accuracy()

    def svm_fit(self):
        self.clf.fit(self.xtrain, self.ytrain.values.ravel())

    def svm_predict(self):
        self.predictions = [int(a) for a in self.clf.predict(self.xtest)]

    def svm_accuracy(self):
        self.num_correct = sum(int(a == y) for a, y in zip(self.predictions, self.ytest.values.ravel()))
        self.logger.info("[svm] %s of %s = %s" % (self.num_correct, len(self.ytest), self.num_correct/float(len(self.ytest))))

    
if __name__ == "__main__":
#temp driver
#TODO create driver - controller py
    CSV_FILE = '~/store/fraud_data/creditcard.csv'
    YCOL = 'Class'
    logger = Logging()
    m = Model()
    proc = Processor()
    sv = SVM()


#TODO make this test suite
    data = proc.load_csv(CSV_FILE)
    data = proc.normalize_col(data, 'Amount')
    data = data.drop(['Time'],axis=1)
    X = proc.get_xvals(data, YCOL)
    y = proc.get_yvals(data, YCOL)
#print data.describe()
    Xu, yu = proc.under_sample(data, YCOL)
    Xu_train, Xu_test, yu_train, yu_test = proc.cross_validation_sets(Xu, yu,.3,0)
    X_train, X_test, y_train, y_test = proc.cross_validation_sets(X, y,.3,0)
#try this with under sampled data
    sv.svm_run(Xu_train, Xu_test, yu_train, yu_test)
#try this with regular data
    #sv.svm_base(X_train, X_test, y_train, y_test)
