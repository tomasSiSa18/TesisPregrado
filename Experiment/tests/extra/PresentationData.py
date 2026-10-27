import numpy as np
import pennylane as qp
from tabulate import tabulate

from Experiment.classes import Encoder, NoiseMechanisms

device = qp.device('default.mixed', wires=7)
circuit = qp.QNode(Encoder.encodeOneCol, device)

row = {"collateral_enc": 3.0}

rho = circuit(row, 0)
rho_gad = NoiseMechanisms.GADNoiseMultiQubit(rho, 0.5, 0.5)
print("Matriz de densidad de collateral = 3")
print("Sin ruido")
print(tabulate(np.round(rho, 3).astype(str), tablefmt="grid"))

print("Con ruido")
print(tabulate(np.round(rho_gad, 3).astype(str), tablefmt="grid"))

print(f"Status accound min max {Encoder.min_max(1.0, 0, 3)}")
print(f"Month min max {Encoder.min_max(24.0, 4, 72)}")