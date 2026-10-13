import itertools
import math
import pennylane as qp
from Experiment.classes import ProportionalDistance, NoiseMechanisms
from tabulate import tabulate

device = qp.device('default.mixed', wires=1)

k_list = [4, 3, 5, 10]

probability = 0.5
accuracy = 0.001

@qp.qnode(device)
def state(angle):
    qp.H(wires=0)
    if angle != 0:
        qp.RZ(angle,wires=0)
        
    return qp.density_matrix(wires=0)
        
def getE(aState, bState, acc):
    return max(ProportionalDistance.dPD(aState, bState, acc), ProportionalDistance.dPD(bState, aState, acc))

results = []

for k in k_list:
    s_list = []
    s_n_list = []
    max_pair = None
    max_e_exp = -1
    max_e_teo = -1
    for ang in range(k):
        new_state = state((2*math.pi*ang)/k)
        s_list.append(new_state)
        s_n_list.append(NoiseMechanisms.DepolarizingNoise(new_state, probability))
    
    
    combinations = itertools.combinations(range(k), 2)
    
    for (i, j) in combinations:
        d = qp.math.trace_distance(s_list[i], s_list[j])
        e_exp = getE(s_n_list[i], s_n_list[j], accuracy)
        e_teo = NoiseMechanisms.DepolarizingNoiseTeo(probability, d, 2)
        assert(e_exp <= e_teo + 0.002)
        
        if e_exp > max_e_exp:
            max_e_teo = e_teo
            max_e_exp = e_exp
            max_pair = (i,j)
    
    
    delta_ang = abs(((360*max_pair[0])/k)- ((360*max_pair[1])/k))
    max_angle = min(360-delta_ang, delta_ang)
    
    results.append([f"{k}", f"{max_pair[0]}, {max_pair[1]}", f"{max_angle}", f"{max_e_exp}", f"{max_e_teo}"])


cols = ["K", "(i,j)", "Angulo", "Max Ep_exp", "Max Ep_teo"]

print(tabulate(results, cols))        
        


