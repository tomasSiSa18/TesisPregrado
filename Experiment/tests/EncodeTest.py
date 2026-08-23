import pandas as pd
import pennylane as qp
import matplotlib as plt
import numpy as np

from Experiment.classes.Encoder import Encoder

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

device = qp.device('default.mixed', wires=7)

encoding_circuit = Encoder(device, dataset)
test_dict = {'status_account_ord': 1, 'month_duration': 24, 'credit_amount': 3349, 'years_employment_ord': 1, 'status_savings_ord': 3, 'target': 'bad', 'collateral_encoded': 3, 'collateral_enc': 3.0, 'purpose_enc': 5.0, 'credit_history_enc': 0.0, 'housing_enc': 0.0}
#fig, _ = qp.draw_mpl(encoding_circuit.circuit)(test_dict)
#fig.savefig("./Experiment/imgs/test.png")

rho = encoding_circuit.circuit(test_dict)
print(np.round(rho, 3))