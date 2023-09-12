from math import exp
import torch

class PriceModel:
    def __init__(self, S_0, strike, time_to_expiry, implied_vol, dividend, riskfree_rate, option_sign, model_name):
        self.model_name = model_name
        self.S_0 = torch.tensor([S_0], requires_grad=True)
        self.K = torch.tensor([strike], requires_grad=True)
        self.tte = torch.tensor([time_to_expiry], requires_grad=True)
        self.sigma = torch.tensor([implied_vol], requires_grad=True)
        self.q = torch.tensor([dividend], requires_grad=True)
        self.r = torch.tensor([riskfree_rate], requires_grad=True)
        self.option_sign = torch.tensor([option_sign], requires_grad=True)
        
        self._ert = None
        self._ebrt = None
        self._sqrt_tte = None
        self._moneyness = None
        self.greeks = None
        self.gradient = None

        
        # self.model_type = model_type
        
    def __repr__(self):
        out = ('{}, S_0: {}, Strike: {}, time_to_expiry: {}, implied_vol: {}, dividend: {}, riskfree_rate: {}, option_sign: {}'
               .format(self.model_name, self.S_0, self.K, self.tte, self.sigma, 
                       self.q, self.r, self.option_sign)
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
            self._erbt = torch.exp((self.q - self.r) * self.tte)
        return self._erbt

    @property
    def ert(self):
        if self._ert is None:
            self._ert = torch.exp(-self.r * self.tte)
        return self._ert
    
    @property
    def moneyness(self):
        if self._moneyness is None:
            self._moneyness = self.S_0 / self.K
        return self._moneyness

    @property
    def derivative(self):
        if self.greeks is None:
            if self.gradient is None:
                self.gradient = torch.autograd.grad(self.npv_pytorch, self.S_0, create_graph=True)
            self.delta, =  self.gradient
            self.delta.backward(retain_graph=True)
            self.gamma = self.S_0.grad.item()
            self.rho = self.r.grad.item()
            self.vega = self.sigma.grad.item()
            self.theta = self.tte.grad.item()
            self.strike_greek = self.K.grad.item()
            self.greeks = {
                'underlying': self.S_0.item(),
                'strike': self.K.item(),
                'premium': self.npv_pytorch.item(),
                'delta': self.delta.item(),
                'gamma': self.gamma,
                'rho': self.rho, 
                'vega': self.vega, 
                'theta': self.theta, 
                'strike_greek': self.strike_greek,
            }
        return self.greeks
    
    
    

    def is_unexpired(self):
        return self.tte > 0