import torch
from option_pricers.black import black76_torch
from option_pricers.pricing import PriceModel

class kirk76_torch(PriceModel):
        def __init__(self, S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign, S_02, implied2, coor):
            super().__init__(S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign, 'Kirk76')
            
            # self.gradient = None
            self.npv_pytorch = None
            # self.greeks = None
            self.S_02 = torch.tensor([S_02], requires_grad=True)
            self.sigma2 = torch.tensor([implied2], requires_grad=True)
            self.corr = torch.tensor([coor], requires_grad=True)
            self.min = torch.tensor([0.0], requires_grad=True)

        # def premium(self):
        
        #     fs1 = self.S_0 / (self.S_02 + self.K)
        #     fs2 = self.S_02 / (self.S_02  + self.K)
        #     v = torch.sqrt((self.sigma ** 2) + ((self.sigma2 * fs2) ** 2) - (2 * self.corr * self.sigma * self.sigma2 * fs2))
            
        #     option = {
        #         'S_0': fs1,
        #         'strike': 1.,
        #         'time_to_expiry': self.tte,
        #         'implied_vol': v,
        #         'dividend': 0.0,
        #         'riskfree_rate': self.r,
        #         'option_sign': self.option_sign,
        #     }
        #     self.bs76 = black76_torch(**option)
        #     # self.derivative = self.bs76.derivative()
        #     # prem = bs76.premium()

        #     # Have the GBS function return a value
        #     if self.option_sign == 1.:
        #         return self.bs76.premium() * (self.S_02 + self.K)
        #     else:
        #         return self.bs76.premium() * + self.ert*(fs2 + self.K - fs1)
        
        def premium(self):

            Phi = torch.distributions.Normal(0,1).cdf
            F1 = self.S_0/(self.S_02 + self.K)
            F2 = self.S_02/(self.S_02 + self.K)  

            sqrt_var_eff = self.sigma2 * F2
            sqrt_var_ = (torch.sqrt((self.sigma ** 2) + (sqrt_var_eff ** 2) - 
                                    (2 * self.corr * self.sigma * sqrt_var_eff)))
            sqrt_var = sqrt_var_ * torch.sqrt(self.tte)

            d1 = (torch.log(F1)/sqrt_var) + sqrt_var / 2
            d2 = d1 - sqrt_var

            if self.option_sign == 1.:
                undiscounted_calls =  torch.where(sqrt_var > 0,(self.S_0 * Phi(d1) - (self.S_02 + self.K) * Phi(d2)),
                                                  torch.max(self.S_0 - self.S_02 - self.K,self.min))
                self.npv_pytorch =  undiscounted_calls * self.ert
                return self.npv_pytorch
            else:
                undiscounted_puts = torch.where(sqrt_var > 0, (self.S_02 + self.K) * Phi(-d2) - self.S_0 * Phi(-d1),
                                                torch.max(self.S_02 + self.K - self.S_0, self.min))
                self.npv_pytorch =  undiscounted_puts * self.ert
                return self.npv_pytorch
