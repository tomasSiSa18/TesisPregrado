import pennylane as qp
from tabulate import tabulate
from Experiment.classes.LocalEpsilonExperiment import LocalEpsilonCategoricalExperiment
import math
from Experiment.classes.NoiseMechanisms import NoiseType

k_list = [4, 3, 5, 10]

probability = 0.5
accuracy = 0.001

results = []

device = qp.device('default.mixed', wires=1)

@qp.qnode(device)
def state(angle):
    qp.H(wires=0)
    if angle != 0:
        qp.RZ(angle,wires=0)
        
    return qp.density_matrix(wires=0)

for k in k_list:
    
    stateList = []

    for ang in range(k):
        new_state = state((2*math.pi*ang)/k)
        stateList.append(new_state)
    
    
    experimentalEpsilon, theoreticalEpsilon, epsilonList, distanceList, distanceNoiseList, bestPair, delta_ang, max_angle = LocalEpsilonCategoricalExperiment(stateList, NoiseType.GAD, {"p": 0.5, "g": 0.5, "l": 0.25}, 0.001, 1e-9)
    
    results.append([f"{k}", f"{bestPair[0]}, {bestPair[1]}", f"{max_angle}", f"{experimentalEpsilon}", f"{theoreticalEpsilon}"])


cols = ["K", "(i,j)", "Angulo", "Max Ep_exp", "Max Ep_teo"]

print(tabulate(results, cols))        
        


