import pandas as pd
import pennylane as qp
import matplotlib as plt
import numpy as np

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses
from Experiment.classes.Encoder import Encoder
from tqdm import tqdm

#Creo el dispositivo cuantico
device = qp.device('default.mixed', wires=7)

#Leo el German Credit Dataset
dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

#Creo una lista para todas las distancias de traza d entre rho y sigma
d_list = []

#Creo el circuito de codificacion
encoding_circuit = Encoder(device, dataset)

#Codifico D
Qd = encoding_circuit.encodeAll()
rho = QuantProcesses.Aggregate(Qd)
rho_dep = NoiseMechanisms.DepolarizingNoise(rho, 0.6)

#Codifico
total_max = 0
for i in tqdm(range(1000)):
    Qd_prime = encoding_circuit.encodeExcludeOne(i, Qd)
    sigma = QuantProcesses.Aggregate(Qd_prime)
    sigma_dep = NoiseMechanisms.DepolarizingNoise(sigma, 0.6)
    d_list.append(qp.math.trace_distance(rho, sigma))
    rho_to_sigma_dPD = ProportionalDistance.dPD(rho_dep, sigma_dep, 0.001, i)
    sigma_to_rho_dPD = ProportionalDistance.dPD(sigma_dep, rho_dep, 0.001, i)
    
    iter_max = max(rho_to_sigma_dPD, sigma_to_rho_dPD)
    
    if iter_max > total_max:
        total_max = iter_max

max_trace = max(d_list)

print(f"The theorical epsilon is: {NoiseMechanisms.DepolarizingNoiseTeo(0.6, max_trace, 128)}")
print(f"The experimental epsilon is:{total_max}")
