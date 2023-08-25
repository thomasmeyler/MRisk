    
import torch
 
class blackScholes_pyTorch(PriceModel):
    def __init__(self, S_0, strike, time_to_expiry, implied_vol, riskfree_rate, option_sign):
        pass
    
  
def blackScholes_pyTorch( S_0, strike, time_to_expiry, implied_vol, riskfree_rate):
        super().__init__(s, k, r, b, sigma, tte, option_sign, 'blackscholes')
        self._d1 = None
        self._d2 = None
        self._nd1 = None
        self._nd2 = None
        self._nqd = None
        self._nd1d = None
        self._sigma_sqrt_tte = None
    S = S_0
    K = strike
    dt = time_to_expiry
    sigma = implied_vol
    r = riskfree_rate
    Phi = torch.distributions.Normal(0,1).cdf
    d_1 = (torch.log(S_0 / K) + (r+sigma**2/2)*dt) / (sigma*torch.sqrt(dt))
    d_2 = d_1 - sigma*torch.sqrt(dt)
    return S*Phi(d_1) - K*torch.exp(-r*dt)*Phi(d_2)

