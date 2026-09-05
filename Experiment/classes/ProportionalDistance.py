import numpy as np
import math

def dPD(rho: np.ndarray, sigma: np.ndarray, accuracy: float):

    r = rho.copy()
    s = sigma.copy()

    max: float = 0.0
    lam = math.inf
    
    while abs(max - lam) > accuracy:
        
        lam = max
        
        max_i = None
        for i in range(rho.shape[0]):
            div = r[i][i]/s[i][i]
            if max_i is None:
                max_i = div
            else:
                if div > max_i :
                    max_i = div 
        max = max_i
        
        eta = r.copy() - max_i*s.copy()
        _, P = np.linalg.eigh(eta)
        P_t = np.conjugate(P).T
        
        r = P_t * r * P
        s = P_t * s * P
        
    print("Done") 
    return math.log(max)
            
            
            
            
    