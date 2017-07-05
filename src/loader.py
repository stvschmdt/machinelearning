import pandas as pd
import numpy as np
from yahoo_finance import Share
import matplotlib.pyplot as plt
import time
import math

from symbol import Symbol
from logger import Logging

class Loader():

    def __init__(self, sym, s_date, e_date, path=None):
        self.symbol=sym
        self.d_data = {}
        self.log = Logging()
        if not path:
            s = Symbol(sym, s_date, e_date)
            self.d_data[s.name] = s

    def get_data(self, name=None):
        if not name:
            return self.d_data
        elif name in self.d_data:
            return self.d_data[name]
        else:
            self.log.error("{} not found in loader list".format(name))

    def to_csv(self, path=""):
        return 0

    def to_pandas(self, symbol):
        df_s = pd.DataFrame(symbol.data)
        return df_s

    def run_all_methods(self, symbol):
        return 0;

    def data_to_csv(self, symbol):
        symdata = self.get_data(symbol)
        df_data = self.to_pandas(symdata)
        df_data['Date'] = pd.to_datetime(df_data['Date'])
        #df_data['Adj_Close'] = df_data.apply(pd.to_numeric)
        try:
            df_data = df_data.set_index('Date').sort_index()
            return df_data
        except:
            self.log.error("unable to format {} to csv".format(symbol))
