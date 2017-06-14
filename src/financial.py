import os
import numpy as np
import pandas as pd
import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import csv

from logger import Logging

#TODO variable store filenames, verify everything is stored to self, have read in function to store data as pd/np

class Finance(object):
    def __init__(self, infile, outfile='~/store/fin_data/'):
        self.data = pd.DataFrame()
        #dictionary of filename jpgs starting on the ith day value is y val on the i+n+1 th day
        self.y_vals = {}
        self.infile = infile
        self.outfile = outfile

    def process_files(self, n=30):
        start_time = time.time()
        self.data = pd.read_csv(self.infile)
        self.convert_ndays_to_image(n)
        end_time = time.time()
        print 'deleting temp.png'
        os.remove('/home/ubuntu/store/fin_data/temp.png')
        print 'writing out y_val to y_val.csv'
        self.write_csv(self.y_vals)
        print 'process time: %f'%(end_time - start_time)
        #img = mpimg.imread('temp.jpg')
        #plt.imshow(img)
        #plt.show()
        
    def convert_ndays_to_image(self,n):
        #calulcate the full dataframe values for rolling, std, bollinger bands upfront then parse for imaging
        self.rolling = self.data['Adj_Close'].rolling(n).mean()
        self.std = self.data['Adj_Close'].rolling(n).std()
        self.adj = self.data['Adj_Close']
        self.up_boll = self.rolling + 2* self.std
        self.low_boll = self.rolling - 2* self.std
        #how many batches can be y value labeled (-1) for last
        num_batches = len(self.data) - n - 1
        print 'process begin days %s - %s of %s'%(n, num_batches, len(self.data))
        #starting at nth day (first day with rolling value), ending at len(data)-1 th day for last day with y val
        for i in range(n, num_batches):
            self.write_image(n,i, i+n)
            self.y_vals[i] = self.data['Adj_Close'][i+n+1]

    def write_image(self, n, begin, end):
            #n+1 days in range
            t = np.arange(0,n+1,1)
            plt.plot(t,self.adj.loc[begin:end],t,self.rolling.loc[begin:end],t,self.up_boll.loc[begin:end], t,self.low_boll.loc[begin:end])
            plt.savefig('/home/ubuntu/store/fin_data/temp')
            #just in case - clf
            plt.clf()
            Image.open('/home/ubuntu/store/fin_data/temp.png').save('/home/ubuntu/store/fin_data/%s.jpg'%(begin),'JPEG')

    def write_csv(self,d):
        with open('/home/ubuntu/store/fin_data/y_vals.csv', 'w') as f:
            w = csv.writer(f)
            w.writerows(d.items())

if __name__ == '__main__':
    f = Finance('AAPL.csv')
    f.process_files()

