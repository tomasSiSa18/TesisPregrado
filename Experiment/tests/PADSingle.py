import pandas as pd
import pennylane as qp
import matplotlib.pyplot as plt
import numpy as np

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses, Encoder
from tqdm import tqdm

device = qp.device('default.mixed', wires=7)

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

d_list = []
d_r_list = []
e_list = []

circuit = qp.QNode(Encoder.encodeOne, device)


#Codifico D
Qd = Encoder.encodeAll(dataset, circuit)
rho = QuantProcesses.Aggregate(Qd)
rho_pad = NoiseMechanisms.PhaseAmplitudeNoiseMultiQubit(rho, 0.4, 0.5, 0.6)

#Codifico
for i in tqdm(range(1000)):
    
    Qd_prime = Encoder.encodeExcludeOne(i, Qd)
    sigma = QuantProcesses.Aggregate(Qd_prime)
    sigma_pad = NoiseMechanisms.PhaseAmplitudeNoiseMultiQubit(sigma, 0.4, 0.5, 0.6)
    d_list.append(qp.math.trace_distance(rho, sigma))
    d_r_list.append(qp.math.trace_distance(rho_pad, sigma_pad))
    rho_to_sigma_dPD = ProportionalDistance.dPD(rho_pad.copy(), sigma_pad.copy(), 0.001, i)
    sigma_to_rho_dPD = ProportionalDistance.dPD(sigma_pad.copy(), rho_pad.copy(), 0.001, i)
    
    iter_max = max(rho_to_sigma_dPD, sigma_to_rho_dPD)
    e_list.append(iter_max)
    

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1000), e_list)
axes[0].scatter(np.argmax(e_list), max(e_list), color="red", zorder=3)
axes[0].annotate(f"({np.argmax(e_list)}, {max(e_list):.4f})", (np.argmax(e_list), max(e_list)))
axes[0].set_xlabel("Iteration (excluded record index)")
axes[0].set_ylabel("Epsilon")
axes[0].set_title("Epsilon per iteration")

axes[1].plot(range(1000), d_list)
axes[1].scatter(np.argmax(d_list), max(d_list), color="red", zorder=3)
axes[1].annotate(f"({np.argmax(d_list)}, {max(d_list):.4f})", (np.argmax(d_list), max(d_list)))
axes[1].set_xlabel("Iteration (excluded record index)")
axes[1].set_ylabel("Trace distance")
axes[1].set_title("Trace distance per iteration")

axes[2].plot(range(1000), d_r_list)
axes[2].scatter(np.argmax(d_r_list), max(d_r_list), color="red", zorder=3)
axes[2].annotate(f"({np.argmax(d_r_list)}, {max(d_r_list):.4f})", (np.argmax(d_r_list), max(d_r_list)))
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Trace distance")
axes[2].set_title("Trace distance after noise per iteration")


plt.tight_layout()
plt.show()

max_trace = max(d_list)
max_epsilon = max(e_list)
print(f"The max trace is {max_trace}")
print(f"The theorical epsilon is: {NoiseMechanisms.PhaseAmplitudeNoiseTeo(max_trace, 0.5, 0.4)}")
print(f"The experimental epsilon is:{max_epsilon}")