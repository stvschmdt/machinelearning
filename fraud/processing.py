from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report

from logger import Logging



class Processor(object):
        
    def __init__(self, cmds={}):
        self.commands = cmds
        self.logger = Logging()

    def load_csv(self, filename):
        '''return a pandas dataframe of csv contents
        '''
        df = pd.read_csv(filename)
        return df

    def load_pickle(self, filename):
        '''return a pandas dataframe of pickled  contents
        '''
        df = pd.read_pickle(filename)
        return df

    def normalize_col(self, df, col_name):
        df[col_name] = (df[col_name] - df[col_name].mean())/df[col_name].max()
        return df

    #preprocessing with library call 
    def scikit_preprocess_col(self, df, col_name):
        '''normalize a single column of data
           input: dataframe, name of column
           output: normalized column of data to mean 0 std 1 ish
        '''
        df['col_name'] = StandardScaler().fit_transform(df['col_name'].reshape(-1, 1))
        return df

    def get_xvals(self, df, yvar):
        '''parse dataframe containing both independent and dependent vars
           returns X variables only
        '''
        X = df.ix[:, df.columns != yvar]
        return X

    def get_yvals(self, df, yvar):
        '''parse dataframe containing both independent and dependent vars
           returns y variable only
        '''
        y = df.ix[:, df.columns == yvar]
        return y

    def one_hot_vectorization(self, n, j):
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

    def under_sample(self, df, y_col):
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
        self.logger.info("[proc] normal transactions: %s"%(len(under_sample_df[under_sample_df[y_col] == 0])/len(under_sample_df)))
        self.logger.info("[proc] fraud transactions: %s"%(len(under_sample_df[under_sample_df[y_col] == 1])/len(under_sample_df)))
        self.logger.info("[proc] number of transactions in resample: %s"%len(under_sample_df))
        return X_undersample, y_undersample

    #TODO move this to model
    def cross_validation_sets(self, xvars, yvars, test_split, rand_state):
        X_train, X_test, y_train, y_test = train_test_split(xvars, yvars, test_size=test_split, random_state=rand_state) 
        self.logger.info("[proc] transactions train dataset: %s"%len(X_train))
        self.logger.info("[proc] transactions test dataset: %s"%len(X_test))
        self.logger.info("[proc] total transactions: %s"%(len(X_train)+len(X_test)))
        return X_train, X_test, y_train, y_test
