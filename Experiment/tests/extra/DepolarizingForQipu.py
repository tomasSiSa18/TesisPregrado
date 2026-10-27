import pennylane as qp
import math
from Experiment.classes import NoiseMechanisms

device = qp.device('default.mixed', wires=1)
device_2 = qp.device('default.mixed', wires=2)

@qp.qnode(device)
def encode(value: int, k: int):
    qp.H(wires=0)
    qp.RZ((2*math.pi*value)/k, wires=0)
    return qp.density_matrix(wires=0)

@qp.qnode(device_2)
def gad_exp_1(env_setting):
    if (env_setting == 'one'):
        qp.X(wires=1)
    qp.RX(math.pi/2, wires=0)
    qp.RX(math.pi/2, wires=1)
    qp.CNOT(wires=[0,1])
    qp.RX(0.866, wires=0)
    qp.RX(0.866, wires=1)
    qp.RZ(0.866, wires=0)
    qp.RZ(0.866, wires=1)
    qp.CNOT(wires=[0,1])
    qp.RX(-(math.pi/2), wires=0)
    qp.RX(-(math.pi/2), wires=1)
    return qp.density_matrix(wires=[0,1])
    


def depolarize(density_matrix, method: str):
    
    if(method == 'DEP'):
        return NoiseMechanisms.DepolarizingNoise(density_matrix, 0.6)
    elif(method == 'GAD'):
        return NoiseMechanisms.GADNoiseSingleQubit(density_matrix, 0.5, 0.6, 0, 1)
    elif(method == 'PAD'):
        return NoiseMechanisms.PhaseAmplitudeNoiseSingleQubit(density_matrix, 0.4, 0.6, 0.5, 0, 1)
    
    

density_matrix = encode(0, 4)

#print(depolarize(density_matrix, 'DEP'))

print(gad_exp_1('zero'))

