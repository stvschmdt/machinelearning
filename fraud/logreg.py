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

class LogReg(Model):

    def __init__(self, params={}, debug=0, name='logistic_regrssion'):
        Model.__init__(self, params, debug)
        #self.name = name

    #TODO move this to model
    def printing_Kfold_scores(self, x_train_data,y_train_data):
        fold = KFold(len(y_train_data),5,shuffle=False) 

        # Different C parameters
        c_param_range = [0.01,0.1,1,10,100]

        results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
        results_table['C_parameter'] = c_param_range

        # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
        j = 0
        for c_param in c_param_range:
            logger.results('C parameter: %f'% c_param)

            recall_accs = []
            for iteration, indices in enumerate(fold,start=1):

                # Call the logistic regression model with a certain C parameter
                lr = LogisticRegression(C = c_param, penalty = 'l1')

                # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
                # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
                lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

                # Predict values using the test indices in the training data
                y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

                # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
                recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
                recall_accs.append(recall_acc)
                logger.info('Iteration: %f Recall: %f'%(iteration,recall_acc))

            # The mean value of those recall scores is the metric we want to save and get hold of.
            results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
            j += 1
            logger.results('Mean recall score %f'%np.mean(recall_accs))

        best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
        
        # Finally, we can check which C parameter is the best amongst the chosen.
        logger.info('Best model to choose from cross validation is with C parameter = %f'%best_c)
        
        return best_c

    #TODO move this to model
    def logistic_regression(self, xtrain, xtest, ytrain, ytest, c, penalty='l1'):
        lr = LogisticRegression(C=c, penalty=penalty)
        lr.fit(xtrain,ytrain.values.ravel())
        ypredict = lr.predict(xtest.values)

        # Compute confusion matrix
        cnfmatrix = confusion_matrix(ytest,ypredict)
        np.set_printoptions(precision=2)

        logger.results("Recall metric in the testing dataset: %f"%(cnfmatrix[1,1]/(cnfmatrix[1,0]+cnfmatrix[1,1])))
        return ypredict

    #TODO move this to model class
    def get_roc_curve(self, xtrain, xtest, ytrain, ytest, c, penalty='l1'):
        lr = LogisticRegression(C = c, penalty = 'l1')
        y_pred_score = lr.fit(xtrain, ytrain.values.ravel()).decision_function(xtest.values)

        fpr, tpr, thresholds = roc_curve(ytest.values.ravel(),y_pred_score)
        roc_auc = auc(fpr,tpr)
        #for v in zip(fpr,tpr):
            #print v
        logger.results('AUC: %f'%roc_auc)
        return roc_auc




if __name__ == '__main__':
#temp driver
#TODO create driver - controller py
    CSV_FILE = '~/store/fraud_data/creditcard.csv'
    YCOL = 'Class'
    logger = Logging()
    m = Model()
    p = {'c':.3, 'iters':100}
    proc = Processor()
    lin = LogReg(p)
    print lin.get_parameters()


#TODO make this test suite
    data = proc.load_data(CSV_FILE)
    data = proc.normalize_col(data, 'Amount')
    data = data.drop(['Time'],axis=1)
    X = proc.get_xvals(data, YCOL)
    y = proc.get_yvals(data, YCOL)
#print data.describe()
    Xu, yu = proc.under_sample(data, YCOL)
    Xu_train, Xu_test, yu_train, yu_test = proc.cross_validation_sets(Xu, yu,.3,0)
    X_train, X_test, y_train, y_test = proc.cross_validation_sets(X, y,.3,0)
#try this with under sampled data
    c = lin.printing_Kfold_scores(Xu_train, yu_train)
    lin.logistic_regression(Xu_train, Xu_test, yu_train, yu_test, c)
    lin.logistic_regression(Xu_train, X_test, yu_train, y_test, c)
    lin.get_roc_curve(Xu_train, Xu_test, yu_train, yu_test, c)
#try this with regular data
    c = lin.printing_Kfold_scores(X_train, y_train)
    lin.logistic_regression(X_train, X_test, y_train, y_test, c)
    lin.get_roc_curve(X_train, X_test, y_train, y_test, c)
