import numpy as np
import math

def DepolarizingNoise(density_matrix: np.ndarray, p: float):
        
    D = density_matrix.shape[0]
    identinty_matrix = np.identity(D)
    
    return (p*identinty_matrix/D) + ((1-p)*density_matrix)

def DepolarizingNoiseTeo(p: float, d: float, D: int):
    return math.log(1+(((1-p)/p) * d * D))

def GADNoiseSingleQubit(density_matrix: np.ndarray, p: float, g: float):
    
    dep_rho = np.zeros_like(density_matrix)
    
    E_0 = math.sqrt(p) * np.array([[1,0],[0, math.sqrt(1-g)]])
    E_1 = math.sqrt(p) * np.array([[0,math.sqrt(g)],[0, 0]])
    E_2 = math.sqrt(1 - p) * np.array([[math.sqrt(1-g),0],[0, 1]])
    E_3 = math.sqrt(1 - p) * np.array([[0,0],[math.sqrt(g), 0]]) 
    
    krauss = [E_0, E_1, E_2, E_3]
    
    for E in krauss:
        dep_rho += E @ density_matrix.copy() @ np.conjugate(E.copy()).T  
        
    return dep_rho

def GADNoiseTeo(d: float, g: float):
    return math.log(1+((2*d*math.sqrt(1-g))/(1-math.sqrt(1-g)))) 

def PhaseAmplitudeNoiseSingleQubit(density_matrix: np.ndarray, l: float, g: float, p: float):
    
    E_0 = np.array([[1, 0], [0, math.sqrt(1-l)]])
    E_1 = np.array([[0, 0], [0, math.sqrt(l)]]) 
    
    krauss = [E_0, E_1]
    
    dep_rho = np.zeros_like(density_matrix)
    
    for E in krauss:
        dep_rho += E @ density_matrix.copy() @ np.conjugate(E.copy()).T
        
    return GADNoiseSingleQubit(dep_rho, p, g)

def PhaseAmplitudeNoiseTeo(d: float, g: float, l: float):
    return math.log(1+((2*d*math.sqrt(1-g)*math.sqrt(1-l))/(1-(math.sqrt(1-g)*math.sqrt(1-l)))))