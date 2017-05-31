import numpy as np
import pandas as pd
from logger import Logging
from processing import Processor

class Model(object):

    def __init__(self, params={}, debug=0):
        self.parameters=params
        self.debug=0
        self.num_params=len(params)
        self.info = {}
        self.results = {}
        self.logger = Logging()
        self.processor = Processor()

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, p, val):
        if p in parameters:
            logger.info("setting parameters[%s] to %f"%(p,val))
        self.parameters[p] = val

    def get_info(self, var):
        return self.info.get(var)

    def get_results(self, var):
        return self.results.get(var)
