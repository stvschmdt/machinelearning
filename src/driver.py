from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
import argparse


from logger import Logging
from model import Model
from processing import Processor
from svm import SVM
from logreg import LogReg
from loader import Loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ml driver arguments')
    parser.add_argument('-lr', dest='LR_DRIVE', default=False, action='store_true')
    parser.add_argument('-svm', dest='SVM_DRIVE', default=False, action='store_true')
    parser.add_argument('-svml', dest='SVML_DRIVE', default=False, action='store_true')
    parser.add_argument('-d', dest='CSV_FILE', default='~/store/fraud_data/creditcard.csv')
    parser.add_argument('-yclass', dest='YCOL', default='Class')
    args = parser.parse_args()
    #arguments for running ml suite

    #driver - controller.py
    #CSV_FILE = '~/store/fraud_data/creditcard.csv'
    #YCOL = 'Class'
    logger = Logging()
    m = Model()
    proc = Processor()

    #processor
    data = proc.load_csv(args.CSV_FILE)
    data = proc.normalize_col(data, 'Amount')
    data = data.drop(['Time'],axis=1)
    print data[args.YCOL].value_counts()
    X = proc.get_xvals(data, args.YCOL)
    y = proc.get_yvals(data, args.YCOL)
    
    #processor xfolds
    Xu, yu = proc.under_sample(data, args.YCOL)
    Xu_train, Xu_test, yu_train, yu_test = proc.cross_validation_sets(Xu, yu,.3,0)
    X_train, X_test, y_train, y_test = proc.cross_validation_sets(X, y,.3,0)

    if args.LR_DRIVE:
        lin = LogReg()
        #under sampled data
        c = lin.printing_Kfold_scores(Xu_train, yu_train)
        lin.logistic_regression(Xu_train, Xu_test, yu_train, yu_test, c)
        lin.logistic_regression(Xu_train, X_test, yu_train, y_test, c)
        lin.get_roc_curve(Xu_train, Xu_test, yu_train, yu_test, c)
        #regular data
        c = lin.printing_Kfold_scores(X_train, y_train)
        lin.logistic_regression(X_train, X_test, y_train, y_test, c)
        lin.get_roc_curve(X_train, X_test, y_train, y_test, c)

    if args.SVM_DRIVE:
        sv = SVM()
        sv.svm_run(Xu_train, Xu_test, yu_train, yu_test)
    if args.SVML_DRIVE:
        sv = SVM()
        sv.svm_run(X_train, X_test, y_train, y_test)

    loader = Loader('AAPL','2016-11-01', '2016-11-30')
    aapl = loader.get_data('AAPL')
    print aapl.data
