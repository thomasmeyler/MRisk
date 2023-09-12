

from option_pricers.pricing import PriceModel
import torch
 

class black76_torch(PriceModel):
    def __init__(self, S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign):
        super().__init__(S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign, 'Black76')
        
        self.gradient = None
        self.npv_pytorch = None
        self.greeks = None
    
    def premium(self):