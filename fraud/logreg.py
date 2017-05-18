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

    

def load_data(filename):
    '''return a pandas dataframe of csv contents
    '''
    df = pd.read_csv(filename)
    return df

def normalize_col(df, col_name):
    df[col_name] = (df[col_name] - df[col_name].mean())/df[col_name].max()
    return df

#preprocessing with library call 
def scikit_preprocess_col(df, col_name):
    '''normalize a single column of data
       input: dataframe, name of column
       output: normalized column of data to mean 0 std 1 ish
    '''
    df['col_name'] = StandardScaler().fit_transform(df['col_name'].reshape(-1, 1))
    return df

def get_xvals(df, yvar):
    '''parse dataframe containing both independent and dependent vars
       returns X variables only
    '''
    X = df.ix[:, df.columns != yvar]
    return X

def get_yvals(df, yvar):
    '''parse dataframe containing both independent and dependent vars
       returns y variable only
    '''
    y = df.ix[:, df.columns == yvar]
    return y

def one_hot_vectorization(n, j):
    '''Return a length n vector with a 1.0 in the jth
    position and zeroes elsewhere. For target y variable.
    '''
    if isinstance(e, list):
        v = np.zeros((len(n), 1))
        v[j] = 1.0
    else:
        v = np.zeros((n, 1))
        v[j] = 1.0
    return v

def under_sample(df, y_col):
    '''under sampling function - unbalanced data set
       input: dataframe and y variable column name
       output: smaller dataframe with 50/50 split of binary classes
       todo: add parameter for which is the minority/majority automatically
    '''
    # Number of df points in the minority class
    number_records_fraud = len(df[df[y_col] == 1])
    fraud_indices = np.array(df[df[y_col] == 1].index)

    # Picking the indices of the normal classes
    normal_indices = df[df[y_col] == 0].index

    # Out of the indices we picked, randomly select "x" number (number_records_fraud)
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
    random_normal_indices = np.array(random_normal_indices)

    # Appending the 2 indices
    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

    # Under sample dfset
    under_sample_df = df.iloc[under_sample_indices,:]

    X_undersample = under_sample_df.ix[:, under_sample_df.columns != y_col]
    y_undersample = under_sample_df.ix[:, under_sample_df.columns == y_col]

    # Showing ratio
    # TODO use logger in debug mode or verbose mode for this
    logger.info("Percentage of normal transactions: %f"%(len(under_sample_df[under_sample_df[y_col] == 0])/len(under_sample_df)))
    logger.info("Percentage of fraud transactions: %f"%(len(under_sample_df[under_sample_df[y_col] == 1])/len(under_sample_df)))
    logger.info("Total number of transactions in resampled df: %f"%len(under_sample_df))
    return X_undersample, y_undersample

#TODO move this to model
def cross_validation_sets(xvars, yvars, test_split, rand_state):
    X_train, X_test, y_train, y_test = train_test_split(xvars, yvars, test_size=test_split, random_state=rand_state) 
    logger.info("Number transactions train dataset: %f"%len(X_train))
    logger.info("Number transactions test dataset: %f"%len(X_test))
    logger.info("Total number of transactions: %f"%(len(X_train)+len(X_test)))
    return X_train, X_test, y_train, y_test

#TODO move this to model
def printing_Kfold_scores(x_train_data,y_train_data):
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
def logistic_regression(xtrain, xtest, ytrain, ytest, c, penalty='l1'):
    lr = LogisticRegression(C=c, penalty=penalty)
    lr.fit(xtrain,ytrain.values.ravel())
    ypredict = lr.predict(xtest.values)

    # Compute confusion matrix
    cnfmatrix = confusion_matrix(ytest,ypredict)
    np.set_printoptions(precision=2)

    logger.results("Recall metric in the testing dataset: %f"%(cnfmatrix[1,1]/(cnfmatrix[1,0]+cnfmatrix[1,1])))
    return ypredict

#TODO move this to model class
def get_roc_curve(xtrain, xtest, ytrain, ytest, c, penalty='l1'):
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
    CSV_FILE = 'data/creditcard.csv'
    YCOL = 'Class'
    logger = Logging()


#TODO make this test suite
    data = load_data(CSV_FILE)
    data = normalize_col(data, 'Amount')
    data = data.drop(['Time'],axis=1)
    X = get_xvals(data, YCOL)
    y = get_yvals(data, YCOL)
#print data.describe()
    Xu, yu = under_sample(data, YCOL)
    Xu_train, Xu_test, yu_train, yu_test = cross_validation_sets(Xu, yu,.3,0)
    X_train, X_test, y_train, y_test = cross_validation_sets(X, y,.3,0)
#try this with under sampled data
    c = printing_Kfold_scores(Xu_train, yu_train)
    logistic_regression(Xu_train, Xu_test, yu_train, yu_test, c)
    logistic_regression(Xu_train, X_test, yu_train, y_test, c)
    get_roc_curve(Xu_train, Xu_test, yu_train, yu_test, c)
#try this with regular data
    c = printing_Kfold_scores(X_train, y_train)
    logistic_regression(X_train, X_test, y_train, y_test, c)
    get_roc_curve(X_train, X_test, y_train, y_test, c)
    m = Model()
