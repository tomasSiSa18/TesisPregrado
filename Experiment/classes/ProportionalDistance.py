import numpy as np
import math

def dPD(rho: np.ndarray, sigma: np.ndarray, accuracy: float, iter: int = None):

    r = rho.copy()
    s = sigma.copy()

    max_ratio: float = 0.0
    lam = math.inf
    
    while abs(max_ratio - lam) > accuracy:
        
        lam = max_ratio
        
        max_i = None
        for i in range(rho.shape[0]):
            div = float(r[i][i].real)/float(s[i][i].real)
            if max_i is None:
                max_i = div
            else:
                if div > max_i :
                    max_i = div 
        max_ratio = max_i
        
        eta = r.copy() - max_ratio * s.copy()
        _, P = np.linalg.eigh(eta)
        P_t = np.conjugate(P).T
        
        r = P_t @ r @ P
        s = P_t @ s @ P
        
    if iter is not None:
        print(f"Iteration {iter} DONE")
    return math.log(max_ratio)
            
            
            
            
    