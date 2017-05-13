from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report


def load_data(filename):
	df = pd.read_csv(filename)
	return df

def normalize_col(df, col_name):
	df[col_name] = (df[col_name] - df[col_name].mean())/df[col_name].max()
	return df

def scikit_preprocess_col(df, col_name):
	df['col_name'] = StandardScaler().fit_transform(df['col_name'].reshape(-1, 1))
	return df

def get_xvals(df, yvar):
        X = df.ix[:, df.columns != yvar]
	return X

def get_yvals(df, yvar):
        y = df.ix[:, df.columns == yvar]
	return y

def under_sample(df, y_col):
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
	print "Percentage of normal transactions: ", len(under_sample_df[under_sample_df[y_col] == 0])/len(under_sample_df)
	print "Percentage of fraud transactions: ", len(under_sample_df[under_sample_df[y_col] == 1])/len(under_sample_df)
	print "Total number of transactions in resampled df: ", len(under_sample_df)
	return X_undersample, y_undersample

def cross_validation_sets(xvars, yvars, test_split, rand_state):
	X_train, X_test, y_train, y_test = train_test_split(xvars, yvars, test_size=test_split, random_state=rand_state) 
	print("Number transactions train dataset: ", len(X_train))
	print("Number transactions test dataset: ", len(X_test))
	print("Total number of transactions: ", len(X_train)+len(X_test))
	return X_train, X_test, y_train, y_test

def printing_Kfold_scores(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 

    # Different C parameters
    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

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
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c


def logistic_regression(xtrain, xtest, ytrain, ytest, c, penalty='l1'):
	lr = LogisticRegression(C=c, penalty=penalty)
	lr.fit(xtrain,ytrain.values.ravel())
	ypredict = lr.predict(xtest.values)

	# Compute confusion matrix
	cnfmatrix = confusion_matrix(ytest,ypredict)
	np.set_printoptions(precision=2)

	print "Recall metric in the testing dataset: ", cnfmatrix[1,1]/(cnfmatrix[1,0]+cnfmatrix[1,1])
	return ypredict

def get_roc_curve(xtrain, xtest, ytrain, ytest, c, penalty='l1'):
	lr = LogisticRegression(C = c, penalty = 'l1')
	y_pred_score = lr.fit(xtrain, ytrain.values.ravel()).decision_function(xtest.values)

	fpr, tpr, thresholds = roc_curve(ytest.values.ravel(),y_pred_score)
	roc_auc = auc(fpr,tpr)
	#for v in zip(fpr,tpr):
		#print v
	print 'AUC:',roc_auc
	return roc_auc


#temp driver
CSV_FILE = 'creditcard.csv'
YCOL = 'Class'

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
