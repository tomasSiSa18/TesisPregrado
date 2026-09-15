import numpy as np
import math

def DepolarizingNoise(density_matrix: np.ndarray, p: float):
        
    D = density_matrix.shape[0]
    identinty_matrix = np.identity(D)
    
    return (p*identinty_matrix/D) + ((1-p)*density_matrix)

def DepolarizingNoiseTeo(p: float, d: float, D: int):
    return math.log(1+(((1-p)/p) * d * D))

def getTensorEtoRho(E: np.ndarray, D: int, wire: int, num_qbits: int):
    
    I = np.identity(2)
    if wire == 0:
        result = E.copy()
    else:
        result = I.copy()
    
    for i in range(1, num_qbits):
        
        if i == wire:
            result = np.kron(result.copy(), E.copy())
        else:
            result = np.kron(result.copy(), I.copy())
    
    return result

def GADNoiseSingleQubit(density_matrix: np.ndarray, p: float, g: float, wire: int, D: int, num_qbits: int):
    
    dep_rho = np.zeros_like(density_matrix)
    
    E_0 = math.sqrt(p) * np.array([[1,0],[0, math.sqrt(1-g)]])
    E_1 = math.sqrt(p) * np.array([[0,math.sqrt(g)],[0, 0]])
    E_2 = math.sqrt(1 - p) * np.array([[math.sqrt(1-g),0],[0, 1]])
    E_3 = math.sqrt(1 - p) * np.array([[0,0],[math.sqrt(g), 0]]) 
    
    krauss = [E_0, E_1, E_2, E_3]
    
    for E in krauss:
        tensor_E = getTensorEtoRho(E, D, wire, num_qbits)
        dep_rho += tensor_E.copy() @ density_matrix.copy() @ np.conjugate(tensor_E.copy()).T  
        
    return dep_rho

def GADNoiseMultiQubit(density_matrix: np.ndarray, p: float, g: float):
    
    D = density_matrix.shape[0]
    num_qbits = round(math.log2(D))
    result = density_matrix.copy()
    
    for wire in range(num_qbits):
        result = GADNoiseSingleQubit(result.copy(), p, g, wire, D, num_qbits)
        
    return result

def GADNoiseTeo(d: float, g: float):
    return math.log(1+((2*d*math.sqrt(1-g))/(1-math.sqrt(1-g)))) 

def PhaseAmplitudeNoiseSingleQubit(density_matrix: np.ndarray, l: float, g: float, p: float, wire: int, D: int, num_qbits: int):
    
    E_0 = np.array([[1, 0], [0, math.sqrt(1-l)]])
    E_1 = np.array([[0, 0], [0, math.sqrt(l)]]) 
    
    krauss = [E_0, E_1]
    
    dep_rho = np.zeros_like(density_matrix)
    
    for E in krauss:
        tensor_E = getTensorEtoRho(E, D, wire, num_qbits)
        dep_rho += tensor_E.copy() @ density_matrix.copy() @ np.conjugate(tensor_E.copy()).T 
        
    return GADNoiseSingleQubit(dep_rho, p, g)

def PhaseAmplitudeNoiseMultiQubit(density_matrix: np.ndarray, l: float, g: float, p: float):
    
    D = density_matrix.shape[0]
    num_qbits = round(math.log2(D))
    result = density_matrix.copy()
    
    for wire in range(num_qbits):
        result = PhaseAmplitudeNoiseSingleQubit(result.copy(), l, g, p, wire, D, num_qbits)
        
    return result

def PhaseAmplitudeNoiseTeo(d: float, g: float, l: float):
    return math.log(1+((2*d*math.sqrt(1-g)*math.sqrt(1-l))/(1-(math.sqrt(1-g)*math.sqrt(1-l)))))