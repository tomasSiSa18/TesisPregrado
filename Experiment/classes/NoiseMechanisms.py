import numpy as np
import math

def DepolarizingNoise(density_matrix: np.ndarray, p: float):
        
    D = density_matrix.shape[0]
    identinty_matrix = np.identity(D)
    
    return (p*identinty_matrix/D) + ((1-p)*density_matrix)

def DepolarizingNoiseTeo(p: float, d: float, D: int):
    return math.log(1+(((1-p)/p) * d * D))
        