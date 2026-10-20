import pandas as pd
import pennylane as qp
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses, Encoder
from tqdm import tqdm

device = qp.device('default.mixed', wires=1)

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")


max_epsilon=0
max_trace = 0
teo_ep = 0
epsilons = {}
cols=["collateral_enc", "purpose_enc", "credit_history_enc", "housing_enc", "status_account_ord and month_duration", "status_savings_ord and years_employment_ord", "credit_amount"]

for k in tqdm(range(7)):
    
    d_list = []
    e_list = []

    dict_df = dataset.to_dict(orient="records")
    circuit = qp.QNode(Encoder.encodeOneCol, device)

    #Codifico D
    Qd = Encoder.encodeAllSingle(dataset, circuit, k)
    rho = QuantProcesses.Aggregate(Qd)
    rho_gad = NoiseMechanisms.DepolarizingNoise(rho, 0.5)

    #Codifico
    for i in tqdm(range(1000)):
        
        Qd_prime = Encoder.encodeExcludeOne(i, Qd)
        sigma = QuantProcesses.Aggregate(Qd_prime)
        sigma_gad = NoiseMechanisms.DepolarizingNoise(sigma, 0.5)
        d_list.append(qp.math.trace_distance(rho, sigma))
        rho_to_sigma_dPD = ProportionalDistance.dPD(rho_gad.copy(), sigma_gad.copy(), 0.001, i)
        sigma_to_rho_dPD = ProportionalDistance.dPD(sigma_gad.copy(), rho_gad.copy(), 0.001, i)
        
        iter_max = max(rho_to_sigma_dPD, sigma_to_rho_dPD)
        e_list.append(iter_max)
    
    
    max_trace = max(d_list)
    teo_run = NoiseMechanisms.DepolarizingNoiseTeo(0.5, max_trace, 2)
    teo_ep += teo_run
    
    exp_run = max(e_list)
    max_epsilon += exp_run
    
    epsilons[cols[k]] = [cols[k],str(max_trace),str(teo_run), str(exp_run),  str(teo_run/exp_run)]
    
    
colsTabulate = ["Variable", "Max Trace", "E Teo", "E Exp", "Ratio"]
print(f"Teo {teo_ep}")
print(f"Exp {max_epsilon}")
print(tabulate(epsilons.values(), tablefmt="grid", headers=colsTabulate))
