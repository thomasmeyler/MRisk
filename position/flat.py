import numpy as np
import pandas as pd
import torch

class flat:
    def __init__(self):
        pass
    
    def mcVaR(self, returns,alpha):
        return np.percentile(returns,alpha)

    def mcCVaR(self, returns,alpha):
        belowVar = returns[returns <= self.mcVaR(returns,alpha)]
        return np.mean(belowVar)
    
    @staticmethod
    def returns_weights(prices,dict_conts):
        
        weights = [_ for _ in dict_conts.values()]
        weights_adjuster = [1 if x > 0 else -1 for x in weights] 

        returns = prices.copy()
        returns = returns.multiply(weights_adjuster)

        meanReturns = returns.mean()
        covMatrix = returns.cov()

        weightsA = list(np.array(weights_adjuster) * np.array(weights))

        weightsA /= np.sum(weightsA)
        returns['portfolio'] = returns.dot(weightsA)
        
        return returns, weights, weightsA, weights_adjuster, meanReturns, covMatrix
    
    @staticmethod
    def MCsimulation(n_runs,num_days,vStart,weights_adjuster,lot_multiplier,weightsA,meanReturns,covMatrix):
        
        mc_sims = n_runs # number of simulations
        T = num_days #timeframe in days
        meanM = np.full(shape=(T, len(weightsA)), fill_value=meanReturns)
        meanM = meanM.T
        portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
        initialPortfolio = ((vStart.multiply(weights_adjuster)*lot_multiplier).sum(axis=1)).values[0]
        for m in range(0, mc_sims):
            Z = np.random.normal(size=(T, len(weightsA)))
            L = np.linalg.cholesky(covMatrix)
            dailyReturns = meanM + np.inner(L, Z)
            portfolio_sims[:,m] = np.cumprod(np.inner(weightsA, dailyReturns.T)+1)*initialPortfolio
        portResults = pd.Series(portfolio_sims[-1,:])
        return portResults, portfolio_sims, initialPortfolio
        

class risk_surface():
    def __init__(self):
        pass
    
    @staticmethod
    def shock_map(returns, window,bumps = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]):
        shock_map = returns.copy()
        shock_map = shock_map[-window:].std()
        shock_map = np.array(shock_map)
        shock_map = shock_map.reshape(1,len(shock_map))
        shock_map = np.repeat(shock_map, len(bumps), axis=0)
        shock_map = shock_map*bumps
        
        return shock_map
    
    def blackScholes_pyTorch(self, S_0, strike, time_to_expiry, implied_vol, riskfree_rate):
        S = S_0
        K = strike
        dt = time_to_expiry
        sigma = implied_vol
        r = riskfree_rate
        Phi = torch.distributions.Normal(0,1).cdf
        d_1 = (torch.log(S_0 / K) + (r+sigma**2/2)*dt) / (sigma*torch.sqrt(dt))
        d_2 = d_1 - sigma*torch.sqrt(dt)
        return S*Phi(d_1) - K*torch.exp(-r*dt)*Phi(d_2)

    @staticmethod
    def quick_black(self):
        pass