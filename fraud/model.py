import numpy as np
import pandas as pd
from logger import Logging

class Model(object):

    def __init__(self, params={}, debug=0):
        self.paramaters=params
        self.debug=0
        self.num_params=len(params)
        self.logger = Logging()
        self.info = {}
        self.results = {}

    def get_paramaters(self):
        return self.parameters

    def set_parameters(self, p, val):
        if p in parameters:
            logger.info("setting parameters[%s] to %f"%(p,val))
        self.parameters[p] = val

    def get_info(self, var):
        return self.info.get(var)

    def get_results(self, var):
        return self.results.get(var)
