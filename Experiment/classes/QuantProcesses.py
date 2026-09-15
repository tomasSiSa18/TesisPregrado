import numpy as np

def Aggregate(density_matrices: np.ndarray) -> np.ndarray:
    
    D = density_matrices.shape[1]
    rho = np.zeros((D,D), dtype=complex)
    for matrix in density_matrices:
        rho += matrix
    
    rho /= density_matrices.shape[0]
    
    return rho
        
        