import pandas as pd
import pennylane as qp
import matplotlib as plt
import numpy as np

from Experiment.classes.Encoder import Encoder
from Experiment.classes.NoiseMechanisms import NoiseMechanisms
from Experiment.classes.ProportionalDistance import ProportionalDistance
from Experiment.classes.QuantProcess import QuantProcess

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

device = qp.device('default.mixed', wires=7)

encoding_circuit = Encoder(device, dataset)
process = QuantProcess()
noise = NoiseMechanisms()
proportional_distance = ProportionalDistance(0.1)

Qa = encoding_circuit.encodeAll()
Qb = encoding_circuit.encodeExcludeOne(0, Qa)
rho = process.Aggregate(Qa)
sigma = process.Aggregate(Qb)
print(qp.math.trace_distance(rho, sigma))
rho_dep, sigma_dep = noise.DepolarizingNoise(rho, sigma, 0.6)
proportional_distance.dPD(rho_dep, sigma_dep)
