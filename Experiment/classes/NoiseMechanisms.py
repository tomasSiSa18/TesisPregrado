import numpy as np

def DepolarizingNoise(density_matrix: np.ndarray, p: float):
        
    D = density_matrix.shape[0]
    identinty_matrix = np.identity(D)
    
    return (p*identinty_matrix/D) + ((1-p)*density_matrix)
        