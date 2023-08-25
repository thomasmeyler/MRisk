__author__ = 'tmeyler'

import torch

class PriceModel:
    def __init__(self, S_0, strike, time_to_expiry, implied_vol, b, riskfree_rate, option_sign, model_name):
        self.model_name = model_name
        self.S_0 = S_0
        self.K = strike
        self.tte = time_to_expiry
        self.sigma = implied_vol
        self.b = b
        self.r = riskfree_rate
        self.option_sign = option_sign
        # self.model_type = model_type
        
    def __repr__(self):
        out = ('{}, UL: {}, K: {}, tte: {}, sigma: {}, b: {}, r: {}, option_sign: {}'
               .format(self.model_name, self.S_0, self.K, self.tte, self.sigma, 
                       self.b, self.r, self.option_sign)
            )
        return out
    
    @property
    def sqrt_tte(self):
        if self._sqrt_tte is None:
            self._sqrt_tte = torch.sqrt(self.tte)
        return self._sqrt_tte
    
    
    @property
    def erbt(self):
        if self._erbt is None:
            self._erbt = exp((self.b - self.r) * self.tte)
        return self._erbt

    @property
    def moneyness(self):
        if self._moneyness is None:
            self._moneyness = self.s / self.k
        return self._moneyness

    def is_unexpired(self):
        return self.tte > 0