import numpy as np

class NoiseMechanisms:
    
    def __init__(self):
        return
    
    def depolarize(self, density_matrix: np.ndarray, p: float):
            
        D = density_matrix.shape[0]
        identinty_matrix = np.identity(D)
        
        return (p*identinty_matrix/D) + ((1-p)*density_matrix)
    
    def DepolarizingNoise(self, rho: np.ndarray, sigma: np.ndarray, p: float):
        
        rho_dep = self.depolarize(rho, p)
        sigma_dep = self.depolarize(sigma, p)
        
        return rho_dep, sigma_dep
        