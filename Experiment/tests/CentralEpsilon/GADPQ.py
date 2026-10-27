import pandas as pd
import pennylane as qp
from tabulate import tabulate

from Experiment.classes import Encoder
from Experiment.classes.CentralEpsilonExperiment import CentralEpsilonExperiment, NoiseType
from tqdm import tqdm
device = qp.device('default.mixed', wires=1)

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

experimentalEpsilon=0
max_trace = 0
theoreticalEpsilon = 0
epsilons = {}
cols=["collateral_enc", "purpose_enc", "credit_history_enc", "housing_enc", "status_account_ord and month_duration", "status_savings_ord and years_employment_ord", "credit_amount"]

for k in tqdm(range(7)):
    
    d_list = []
    e_list = []

    dict_df = dataset.to_dict(orient="records")
    circuit = qp.QNode(Encoder.encodeOneCol, device)

    rhoListK = Encoder.encodeAllSingle(dataset, circuit, k)
    
    experimentalEpsilonK, theoreticalEpsilonK, epsilonList, distanceList, _ = CentralEpsilonExperiment(rhoListK, NoiseType.GAD, {"p": 0.5, "g": 0.5}, 0.0000000001)
    
    max_trace = max(distanceList)

    theoreticalEpsilon += theoreticalEpsilonK
    
    experimentalEpsilonK = max(epsilonList)
    experimentalEpsilon += experimentalEpsilonK
    
    epsilons[cols[k]] = [cols[k],str(max_trace),str(theoreticalEpsilonK), str(experimentalEpsilonK),  str(theoreticalEpsilonK/experimentalEpsilonK)]
    
    
colsTabulate = ["Variable", "Max Trace", "E Teo", "E Exp", "Ratio"]
print(f"Teo {theoreticalEpsilon}")
print(f"Exp {experimentalEpsilon}")
print(tabulate(epsilons.values(), tablefmt="grid", headers=colsTabulate))
