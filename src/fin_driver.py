import csv
import numpy as np
from financial import Finance
from reader import Reader
from logger import Logging


if __name__ == '__main__':
    rdr = Reader()
    rdr.read_images('/home/ubuntu/store/fin_data', 'csv',True, False)
    x = np.array(rdr.c_images)
    y = {}
    x_vals = []
    with open('/home/ubuntu/store/fin_data/y_vals.csv', 'r') as f:
        r = csv.reader(f)
        for row in r:
            y[int(row[0])] = float(row[1])
    #print zip(sorted(y),sorted(rdr.file_labels))
    #print sorted(rdr.file_labels)
    x_vals = [ y[int(i)] for i in rdr.file_labels ]
    print x.shape, x_vals

