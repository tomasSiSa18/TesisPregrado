import pandas as pd
import pennylane as qp
import matplotlib.pyplot as plt

from Experiment.classes import Encoder
from tqdm import tqdm
from Experiment.classes.CentralEpsilonExperiment import CentralEpsilonExperiment, NoiseType

#Creo el dispositivo cuantico
device = qp.device('default.mixed', wires=7)

#Leo el German Credit Dataset
dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

e_list_per_p = []
e_teo_per_p = []

probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

circuit = qp.QNode(Encoder.encodeOneRow, device)
rhoListK = Encoder.encodeAll(dataset, circuit)

for p in tqdm(probs):
    
    experimentalEpsilon, theoreticalEpsilon, epsilonList, distanceList, _ = CentralEpsilonExperiment(rhoListK, NoiseType.DEP, {"p": p}, 1e-9)
    
    #Creo una lista para todas las distancias de traza d 
    max_trace = max(distanceList)
    
    e_list_per_p.append(experimentalEpsilon)
    e_teo_per_p.append(theoreticalEpsilon)


ratio_list = []

for i in range(len(probs)):
    if e_list_per_p[i] != 0:
        ratio_list.append(e_teo_per_p[i]/e_list_per_p[i])
    else:
        ratio_list.append(0)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

for p_i, theo, real in zip(probs, e_teo_per_p, e_list_per_p):
    axes[0].plot([p_i, p_i], [theo, real], color="gray", linewidth=1, zorder=1)

axes[0].scatter(probs, e_teo_per_p, color="tab:blue", marker="o", label="Theoretical epsilon", zorder=2)
axes[0].scatter(probs, e_list_per_p, color="tab:orange", marker="s", label="Real epsilon", zorder=2)
axes[0].set_yscale("log")

axes[0].set_xlabel("Probability of Depolarization")
axes[0].set_ylabel("Epsilon")
axes[0].set_title("Theoretical vs real epsilon per probability")
axes[0].legend()

axes[1].plot(probs, ratio_list)
axes[1].set_xlabel("Probabilities")
axes[1].set_ylabel("Teo/Exp ratio")
axes[1].set_title("Teorethical vs Experimental Ratio")

plt.tight_layout()
plt.show()