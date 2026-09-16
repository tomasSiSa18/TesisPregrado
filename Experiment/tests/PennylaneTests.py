from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qp

from Experiment.classes import Encoder, NoiseMechanisms


device = qp.device('default.mixed', wires=2)

@qp.qnode(device)
def bitflip(x):
    qp.X(wires=x)
    return qp.density_matrix([0,1])

#Creo el dispositivo cuantico
device = qp.device('default.mixed', wires=7)

#Leo el German Credit Dataset
dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

def GADtest(row: dict, depolarize: bool):
    Encoder.encodeCategorical(0, row["collateral_enc"], 4.0)
    Encoder.encodeCategorical(1, row["purpose_enc"], 10.0)
    Encoder.encodeCategorical(2, row["credit_history_enc"], 5.0)
    Encoder.encodeCategorical(3, row["housing_enc"], 3.0)
    Encoder.encodeNumerical(4, row["status_account_ord"], row["month_duration"])
    Encoder.encodeNumerical(5, row["status_savings_ord"], row["years_employment_ord"])
    Encoder.encodeNumerical(6, row["credit_amount"])
    if depolarize:
        
        qp.PhaseDamping(0.4, wires=0)
        qp.PhaseDamping(0.4, wires=1)
        qp.PhaseDamping(0.4, wires=2)
        qp.PhaseDamping(0.4, wires=3)
        qp.PhaseDamping(0.4, wires=4)
        qp.PhaseDamping(0.4, wires=5)
        qp.PhaseDamping(0.4, wires=6)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=0)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=1)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=2)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=3)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=4)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=5)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=6)
        
    return qp.density_matrix([0,1,2,3,4,5,6])

'''
if depolarize:
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=0)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=1)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=2)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=3)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=4)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=5)
        qp.GeneralizedAmplitudeDamping(0.5, 0.6, wires=6)

'''

circuit = qp.QNode(GADtest, device)

dict_df = dataset.to_dict(orient="records")

qp.drawer.use_style("black_white")
fig, ax = qp.draw_mpl(circuit)(dict_df[0], True)
plt.show()

matrix = circuit(dict_df[0], False)

dep = NoiseMechanisms.PhaseAmplitudeNoiseMultiQubit(matrix, 0.4, 0.5, 0.6)
#dep = NoiseMechanisms.DepolarizingNoise(matrix, 0.5)

dep_penny = circuit(dict_df[0], True)

print(np.allclose(dep, dep_penny))


