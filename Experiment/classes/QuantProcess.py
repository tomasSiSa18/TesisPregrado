import numpy as np

class QuantProcess:
    
    def __init__(self):
        return 
    
    def Aggregate(self, density_matrices: np.ndarray):
        
        rho = np.zeros((128,128), dtype=complex)
        for matrix in density_matrices:
            rho += matrix
        
        rho /= density_matrices.shape[0]
        
        return rho
        
        