
from option_pricers.pricing import PriceModel
import torch
 
class bachelier_torch(PriceModel):
    def __init__(self, S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign):
        super().__init__(S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign, 'Black76')
        
        # self.gradient = None
        self.npv_pytorch = None
        # self.greeks = None
        # self._cdfd_pdfd = None
        self.bachelier_sigma = self.sigma * self.S_0
    
    # @property
    def cdfd_pdfd(self):
        # if self._cdfd_pdfd is None:
            # Phi = torch.distributions.Normal(0,1)
        d = (self.S_0 - self.K)/(self.bachelier_sigma * self.sqrt_tte)
        cdfd = self.Phi.cdf(self.option_sign * d)
        pdfd = torch.exp(self.Phi.log_prob(d))
        
        return (cdfd, pdfd)

    def premium(self):

        # bachelier_sigma = self.sigma * self.S_0
        cdfd, pdfd = self.cdfd_pdfd()
        self.npv_pytorch = self.ert * cdfd * (self.option_sign * (self.S_0 - self.K)) + \
            pdfd * self.bachelier_sigma * self.sqrt_tte 
        
        return self.npv_pytorch
        
        
        
        
        
        
        
        
        
        
        # d = (self.S_0 * torch.exp(self.r*self.tte) - self.K) / torch.sqrt(bachelier_sigma**2/(2 * self.r) * (torch.exp(2*self.r*self.tte)-1) )

        # C = torch.exp(-self.r * self.tte) * (self.S_0 * torch.exp(self.r * self.tte) - self.K) * Phi.cdf(d) + \
        #         torch.exp(-self.r * self.tte) * torch.sqrt(bachelier_sigma**2/(2*self.r) * (torch.exp(2*self.r*self.tte)-1) ) * torch.exp(Phi.log_prob(d))

        # if self.option_sign == -1.:
        #     self.npv_pytorch = C - self.S_0 + torch.exp(-self.r * self.tte) * self.K 
            
        # else:
        #     self.npv_pytorch = C

        # return self.npv_pytorch

         

    
  
  