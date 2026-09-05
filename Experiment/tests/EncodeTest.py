import pandas as pd
import pennylane as qp
import matplotlib as plt
import numpy as np

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses
from Experiment.classes.Encoder import Encoder

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
for i in range(1):
    Qd_prime = encoding_circuit.encodeExcludeOne(i, Qd)
    sigma = QuantProcesses.Aggregate(Qd_prime)
    sigma_dep = NoiseMechanisms.DepolarizingNoise(sigma, 0.6)
    d_list.append(qp.math.trace_distance(rho, sigma))
    rho_to_sigma_dPD = ProportionalDistance.dPD(rho_dep, sigma_dep)
    sigma_to_rho_dPD = ProportionalDistance.dPD(sigma_dep, rho_dep)
