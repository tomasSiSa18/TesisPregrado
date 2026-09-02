import numpy as np
import math

class ProportionalDistance:
    
    def __init__(self, accuracy: float):
        self.accuracy = accuracy
    
    def dPD(self, rho: np.ndarray, sigma: np.ndarray):

        max: float = 0.0
        lam = math.inf
        
        while abs(max - lam) > self.accuracy:
            
            lam = max
            
            max_i = None
            for i in range(rho.shape[0]):
                div = rho[i][i]/sigma[i][i]
                if max_i is None:
                    max_i = div
                else:
                    if div > max_i :
                        max_i = div 
            
            eta = rho.copy() - max_i*sigma.copy()
            _, P = np.linalg.eigh(eta)
            print()
            P_t = np.conjugate(P).T
            
            
    