from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report

from logger import Logging
from model import Model
from processing import Processor
from svm import SVM
from logreg import LogReg


if __name__ == '__main__':
#temp driver

    LR_DRIVE = 1
    SVM_DRIVE = 1

#TODO create driver - controller py
    CSV_FILE = '~/store/fraud_data/creditcard.csv'
    YCOL = 'Class'
    logger = Logging()
    m = Model()
    proc = Processor()
    lin = LogReg()
    sv = SVM()

    #processor
    data = proc.load_csv(CSV_FILE)
    data = proc.normalize_col(data, 'Amount')
    data = data.drop(['Time'],axis=1)
    X = proc.get_xvals(data, YCOL)
    y = proc.get_yvals(data, YCOL)
    
    #processor xfolds
    Xu, yu = proc.under_sample(data, YCOL)
    Xu_train, Xu_test, yu_train, yu_test = proc.cross_validation_sets(Xu, yu,.3,0)
    X_train, X_test, y_train, y_test = proc.cross_validation_sets(X, y,.3,0)

    if LR_DRIVE:
        #under sampled data
        c = lin.printing_Kfold_scores(Xu_train, yu_train)
        lin.logistic_regression(Xu_train, Xu_test, yu_train, yu_test, c)
        lin.logistic_regression(Xu_train, X_test, yu_train, y_test, c)
        lin.get_roc_curve(Xu_train, Xu_test, yu_train, yu_test, c)
        #regular data
        c = lin.printing_Kfold_scores(X_train, y_train)
        lin.logistic_regression(X_train, X_test, y_train, y_test, c)
        lin.get_roc_curve(X_train, X_test, y_train, y_test, c)

    if SVM_DRIVE:
        sv.svm_run(Xu_train, Xu_test, yu_train, yu_test)
        sv.svm_run(X_train, X_test, y_train, y_test)
