
from option_pricers.pricing import PriceModel
import torch

class black76_torch(PriceModel):
    def __init__(self, S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign):
        super().__init__(S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign, 'Black76')
        
        self.gradient = None
        self.npv_pytorch = None
        self.greeks = None
    
    def premium(self):
         
        Phi = torch.distributions.Normal(0,1).cdf
        
        lnFK = torch.log(self.S_0 / self.K)
        si2 = (self.sigma**2/2)*self.tte
        sqrt_tte = torch.sqrt(self.tte)
        
        d1 = (lnFK + (self.r + si2)) / (self.sigma*sqrt_tte)
        d2 = d1 - self.sigma*sqrt_tte
        
        if self.option_sign == 1.:
            self.npv_pytorch = self.ert * (self.S_0 * Phi(d1) - self.K * Phi(d2))
        else:
            self.npv_pytorch = self.ert * ( self.K * Phi(-d2) - self.S_0 * Phi(-d1) )
        return self.npv_pytorch.item()
    
  
  
  
# options_greeks = []

# for c,v in expiry_dict.items():
#     mapOp = []
#     for K in [3000.,3050.,3100.,3150.,4000.]:
#         res = []
#         for S in t[c]:
#             print(c,K,v)
#             S_0 = torch.tensor([S], requires_grad=True)
#             K = torch.tensor([K], requires_grad=True)
#             T = torch.tensor([v], requires_grad=True)
#             sigma = torch.tensor([0.3], requires_grad=True)
#             r = torch.tensor([0.035], requires_grad=True)

#             npv_pytorch = blackScholes_pyTorch(S_0, K, T, sigma, r)

#             gradient = torch.autograd.grad(npv_pytorch, S_0, create_graph=True)
#             delta, =  gradient
#             delta.backward(retain_graph=True)
        
#             results = {
#                 'S_0': S,
#                 'strike': K.item(),
#                 'price': npv_pytorch.item(),
#                 'rho': r.grad.item(), 
#                 'vega': sigma.grad.item(), 
#                 'theta': T.grad.item(), 
#                 # 'epsilon': 0.0, 
#                 'strike_greek': K.grad.item(),
#                 'delta': delta.item(),
#                 'gamma': S_0.grad.item(),
                
#             }
            
#             res.append(results)
#         res = pd.DataFrame(res)
#         mapOp.append(res)
        
#     options_greeks.append({c: mapOp}) 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
# def blackScholes_pyTorch( S_0, strike, time_to_expiry, implied_vol, riskfree_rate):
#         super().__init__(s, k, r, b, sigma, tte, option_sign, 'blackscholes')
#         self._d1 = None
#         self._d2 = None
#         self._nd1 = None
#         self._nd2 = None
#         self._nqd = None
#         self._nd1d = None
#         self._sigma_sqrt_tte = None
#     S = S_0
#     K = strike
#     dt = time_to_expiry
#     sigma = implied_vol
#     r = riskfree_rate
#     Phi = torch.distributions.Normal(0,1).cdf
#     d_1 = (torch.log(S_0 / K) + (r+sigma**2/2)*dt) / (sigma*torch.sqrt(dt))
#     d_2 = d_1 - sigma*torch.sqrt(dt)
#     return S*Phi(d_1) - K*torch.exp(-r*dt)*Phi(d_2)

