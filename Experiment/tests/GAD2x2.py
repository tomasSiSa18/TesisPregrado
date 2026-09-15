import pandas as pd
import pennylane as qp
import matplotlib.pyplot as plt
import numpy as np

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses, Encoder
from tqdm import tqdm

device = qp.device('default.mixed', wires=1)

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

d_list = []
e_list = []

dict_df = dataset.to_dict(orient="records")
circuit = qp.QNode(Encoder.encodeOneCol, device)

#Codifico D
Qd = Encoder.encodeAll(dataset, circuit)
rho = QuantProcesses.Aggregate(Qd)
rho_gad = NoiseMechanisms.GADNoiseMultiQubit(rho, 0.5, 0.5)

#Codifico
for i in tqdm(range(1000)):
    
    Qd_prime = Encoder.encodeExcludeOne(i, Qd)
    sigma = QuantProcesses.Aggregate(Qd_prime)
    sigma_gad = NoiseMechanisms.GADNoiseMultiQubit(sigma, 0.5, 0.5)
    d_list.append(qp.math.trace_distance(rho, sigma))
    rho_to_sigma_dPD = ProportionalDistance.dPD(rho_gad.copy(), sigma_gad.copy(), 0.001, i)
    sigma_to_rho_dPD = ProportionalDistance.dPD(sigma_gad.copy(), rho_gad.copy(), 0.001, i)
    
    iter_max = max(rho_to_sigma_dPD, sigma_to_rho_dPD)
    e_list.append(iter_max)
    

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(range(1000), e_list)
axes[0].set_xlabel("Iteration (excluded record index)")
axes[0].set_ylabel("Epsilon")
axes[0].set_title("Epsilon per iteration")

axes[1].plot(range(1000), d_list)
axes[1].set_xlabel("Iteration (excluded record index)")
axes[1].set_ylabel("Trace distance")
axes[1].set_title("Trace distance per iteration")

plt.tight_layout()
plt.show()

max_trace = max(d_list)
max_epsilon = max(e_list)
print(f"The max trace is {max_trace}")
print(f"The theorical epsilon is: {NoiseMechanisms.GADNoiseTeo(max_trace, 0.5)}")
print(f"The experimental epsilon is:{max_epsilon}")
