import pandas as pd
import pennylane as qp
import matplotlib.pyplot as plt
import numpy as np

from Experiment.classes import NoiseMechanisms, Encoder
from CentralEpsilonExperiment import CentralEpsilonExperiment, NoiseType

device = qp.device('default.mixed', wires=7)

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

circuit = qp.QNode(Encoder.encodeOneRow, device)

rhoListK = Encoder.encodeAll(dataset, circuit)

experimentalEpsilon, theoreticalEpsilon, epsilonList, distanceList, distanceNoiseList = CentralEpsilonExperiment(rhoListK, NoiseType.DEP, {"p": 0.5}, 0.1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1000), epsilonList)
axes[0].scatter(np.argmax(epsilonList), max(epsilonList), color="red", zorder=3)
axes[0].annotate(f"({np.argmax(epsilonList)}, {max(epsilonList):.4f})", (np.argmax(epsilonList), max(epsilonList)))
axes[0].set_xlabel("Iteration (excluded record index)")
axes[0].set_ylabel("Epsilon")
axes[0].set_title("Epsilon per iteration")

axes[1].plot(range(1000), distanceList)
axes[1].scatter(np.argmax(distanceList), max(distanceList), color="red", zorder=3)
axes[1].annotate(f"({np.argmax(distanceList)}, {max(distanceList):.4f})", (np.argmax(distanceList), max(distanceList)))
axes[1].set_xlabel("Iteration (excluded record index)")
axes[1].set_ylabel("Trace distance")
axes[1].set_title("Trace distance per iteration")

axes[2].plot(range(1000), distanceNoiseList)
axes[2].scatter(np.argmax(distanceNoiseList), max(distanceNoiseList), color="red", zorder=3)
axes[2].annotate(f"({np.argmax(distanceNoiseList)}, {max(distanceNoiseList):.4f})", (np.argmax(distanceNoiseList), max(distanceNoiseList)))
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Trace distance")
axes[2].set_title("Trace distance after noise per iteration")

plt.tight_layout()
plt.show()

max_trace = max(distanceList)
max_epsilon = max(epsilonList)
print(f"The max trace is {max_trace}")
print(f"The theorical epsilon is: {theoreticalEpsilon}")
print(f"The experimental epsilon is:{max_epsilon}")
