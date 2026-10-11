import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qp
from Experiment.classes import ProportionalDistance, NoiseMechanisms

ep_teo_orthogonal = 1.10
ep_teo_neighbor = 0.88

device = qp.device('default.mixed', wires=1)

@qp.qnode(device)
def state(angle):
    qp.H(wires=0)
    if angle != 0:
        qp.RZ(angle,wires=0)
        
    return qp.density_matrix(wires=0)
        
        
plusStateDep = NoiseMechanisms.DepolarizingNoise(state(0), 0.5)
minusStateDep = NoiseMechanisms.DepolarizingNoise(state(math.pi), 0.5)
iStateDep = NoiseMechanisms.DepolarizingNoise(state(math.pi/2), 0.5)

e_exp_orthogonal = max(ProportionalDistance.dPD(plusStateDep, minusStateDep, 0.001), ProportionalDistance.dPD(minusStateDep, plusStateDep, 0.001))
e_exp_neighbor = max(ProportionalDistance.dPD(plusStateDep, iStateDep, 0.001), ProportionalDistance.dPD(iStateDep, plusStateDep, 0.001))

print(e_exp_orthogonal)
print(e_exp_neighbor)



