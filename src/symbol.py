import sys
import math
import time
import csv
import datetime
from yahoo_finance import Share

from logger import Logging

class Symbol(object):

    def __init__(self, symbol, s_date=None, e_date=None):
        self.log=Logging()
        self.name=symbol
        self.created=datetime.datetime.utcnow()
        self.log.info("created {}".format(self.name))
        try:
            self.share=Share(symbol)
        except:
            self.log.error("platform is offline or not connecting")

        if s_date and e_date:
            self.begin=s_date
            self.end=e_date
            try:
                self.share=Share(symbol)
                self.data=self.share.get_historical(self.begin, self.end)
                self.log.refresh("{} data collected".format(self.name))
            except:
                self.log.error("platform is offline or not connecting")
    
    def refresh_data(self, s_date=None, e_date=None):
        if s_date and e_date:
            try:
                share=Share(self.name)
                self.begin=s_date
                self.end=e_date
                share.get_historical(s_date, e_date)
                self.log.refresh("{} data collected".format(self.name))
            except:
                self.log.error("platform is offline or not connecting")

    def market_cap(self):
        try:
            self.market_cap = self.share.get_market_cap()
            self.log.info("{} market cap refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")
        
    def earnings_per_share(self):
        try:
            self.eps = self.share.get_earnings_share()
            self.log.info("{} eps refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")

    def moving_average_50(self):
        try:
            self.moving_average_50 = self.share.get_50day_moving_average()
            self.log.info("{} 50 day moving ave refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")

    #implement TODO
    def nday_moving_average(self, n):
        try:
            self.moving_average_n = None
            self.log.info("{} {} day moving ave refreshed".format(self.name, n))
        except:
            self.log.error("platform is offline or not connecting")

    def price_earnings_ratio(self):
        try:
            self.price_to_earnings = self.share.get_price_earnings_ratio()
            self.log.info("{} price to earnings refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")


    def book_value(self):
        try:
            self.book = self.share.get_price_book()
            self.log.info("{} book value refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")

    def year_high(self):
        try:
            self.year_high = self.share.get_change_from_year_high()
            self.log.info("{} year high change refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")

    def year_low(self):
        try:
            self.year_low = self.share.get_change_from_year_low()
            self.log.info("{} year low change refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")

    def target_price(self):
        try:
            self.year_target = self.share.get_change_from_year_high()
            self.log.info("{} year target change refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")
    
    def year_range(self):
        try:
            self.year_range = self.share.get_change_from_year_high()
            self.log.info("{} year range change refreshed".format(self.name))
        except:
            self.log.error("platform is offline or not connecting")
