import math

import pandas as pd
import pennylane as qp
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses, Encoder
from tqdm import tqdm

probability = 0.5
device = qp.device('default.mixed', wires=1)

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

dict_df = dataset.to_dict(orient="records")
circuit = qp.QNode(Encoder.encodeOneCol, device)

data_encoded = Encoder.encodeAllSingle(dataset, circuit, 0)

plusState = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
plusStateTranspose = plusState.conj().T

probs_normal = []
probs_dep = []

for rho in data_encoded:
    
    probs_normal.append((plusStateTranspose @ rho @ plusState).real)
    
    rho_dep = NoiseMechanisms.DepolarizingNoise(rho, probability)
    
    probs_dep.append((plusStateTranspose @ rho_dep @ plusState).real)

probs_normal = np.array(probs_normal)
probs_dep = np.array(probs_dep)

print(np.unique(np.round(probs_normal, decimals=5)))
print(np.unique(np.round(probs_dep, 5)))

randomGen = np.random.default_rng(8)

uniforms = randomGen.random(size=(2000, 1000))

comparison_normal = uniforms < probs_normal
comparison_dep = uniforms < probs_dep

avg_normal_values = np.mean(comparison_normal, axis=1)
avg_dep_values = np.mean(comparison_dep, axis=1)
avg_dep_corrected_values = (avg_dep_values - (probability/2))/(1-probability)

avg_normal = np.mean(avg_normal_values)
avg_dep = np.mean(avg_dep_values)
avd_dep_corrected = np.mean(avg_dep_corrected_values)

std_normal = np.std(avg_normal_values)
std_dep = np.std(avg_dep_values)
std_corrected = std_dep/(1-probability)

print(f"F_B: Media: {avg_normal} Std: {std_normal}")
print(f"F_C: Media: {avg_dep} Std: {std_dep}")
print(f"F^: Media: {avd_dep_corrected} Std: {std_corrected}")
    
FB = avg_normal_values
FH = avg_dep_corrected_values

ancho = 0.004
bordes = np.arange(min(FB.min(), FH.min()) - 0.0005, max(FB.max(), FH.max()) + ancho, ancho)

plt.hist(FB, bins=bordes, alpha=0.6, label=f"FB, std = {FB.std():.3f}")
plt.hist(FH, bins=bordes, alpha=0.6, label=f"F^, std = {FH.std():.3f}")
plt.axvline(0.45, color="black", linestyle="--", label="valor real")

plt.xlabel("Fraccion de + en cada mundo")
plt.ylabel("Cantidad de muestras")
plt.title(f"Comparativo entre F_B y {r"$\hat{F}$"} para {r"$\varepsilon$"} = 1.10 y p = 0.5 en varios mundos")
plt.legend()
plt.show()
plt.savefig("histograma_collateral")
    
    