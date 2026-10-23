import pennylane as qp
import math
from Experiment.classes import NoiseMechanisms

device = qp.device('default.mixed', wires=1)

@qp.qnode(device)
def encode(value: int, k: int):
    qp.H(wires=0)
    qp.RZ((2*math.pi*value)/k, wires=0)
    return qp.density_matrix(wires=0)


def depolarize(density_matrix, method: str):
    
    if(method == 'DEP'):
        return NoiseMechanisms.DepolarizingNoise(density_matrix, 1)
    elif(method == 'GAD'):
        return NoiseMechanisms.GADNoiseSingleQubit(density_matrix, 0.5, 0.6, 0, 1)
    elif(method == 'PAD'):
        return NoiseMechanisms.PhaseAmplitudeNoiseSingleQubit(density_matrix, 0.4, 0.6, 0.5, 0, 1)
    
    

density_matrix = encode(0, 4)

print(depolarize(density_matrix, 'GAD'))

