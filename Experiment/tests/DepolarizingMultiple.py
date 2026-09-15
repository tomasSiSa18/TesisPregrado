import math

import pandas as pd
import pennylane as qp
import matplotlib.pyplot as plt
import numpy as np

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses
from Experiment.classes.Encoder import Encoder
from tqdm import tqdm

#Creo el dispositivo cuantico
device = qp.device('default.mixed', wires=7)

#Leo el German Credit Dataset
dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

e_list_per_p = []
e_teo_per_p = []

#Creo el circuito de codificacion
encoding_circuit = Encoder(device, dataset)

probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#Codifico D
Qd = encoding_circuit.encodeAll()
rho = QuantProcesses.Aggregate(Qd)
for p in tqdm(probs):
    
    #Creo una lista para todas las distancias de traza d entre rho y sigma
    d_list = []

    #Creo una lista para todos los epsilon
    e_list = []

    rho_dep = NoiseMechanisms.DepolarizingNoise(rho.copy(), p)
    
    for i in tqdm(range(1000)):
        Qd_prime = encoding_circuit.encodeExcludeOne(i, Qd)
        sigma = QuantProcesses.Aggregate(Qd_prime)
        sigma_dep = NoiseMechanisms.DepolarizingNoise(sigma, p)
        d_list.append(qp.math.trace_distance(rho, sigma))
        rho_to_sigma_dPD = ProportionalDistance.dPD(rho_dep.copy(), sigma_dep.copy(), 0.001, i)
        sigma_to_rho_dPD = ProportionalDistance.dPD(sigma_dep.copy(), rho_dep.copy(), 0.001, i)
        
        iter_max = max(rho_to_sigma_dPD, sigma_to_rho_dPD)
        e_list.append(iter_max)
        
    max_trace = max(d_list)
    max_epsilon = max(e_list)
    e_teo = NoiseMechanisms.DepolarizingNoiseTeo(p, max_trace, 128)
    
    e_list_per_p.append(max_epsilon)
    e_teo_per_p.append(e_teo)


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