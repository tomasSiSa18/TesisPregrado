import pandas as pd
import math
import pennylane as qp
import numpy as np
    
def encodeCategorical(wire: int, value: float, k: float):
    qp.H(wires=wire)
    qp.RZ((2*math.pi*value)/k, wires=wire)

def encodeNumerical(wire: int, valueY: int, valueZ: int = 0):
    
    qp.RY(valueY, wires=wire)
    qp.RZ(valueZ, wires=wire)

def encodeOneCol(row: dict, row_num: int):
    if row_num == 0:
        encodeCategorical(0, row["collateral_enc"], 4.0)
    elif row_num == 1:
        encodeCategorical(0, row["purpose_enc"], 10.0)
    elif row_num == 2:
        encodeCategorical(0, row["credit_history_enc"], 5.0)
    elif row_num == 3:
        encodeCategorical(0, row["housing_enc"], 3.0)
    elif row_num == 4:
        encodeNumerical(0, row["status_account_ord"], row["month_duration"])
    elif row_num == 5:
        encodeNumerical(0, row["status_savings_ord"], row["years_employment_ord"])
    elif row_num == 6:
        encodeNumerical(0, row["credit_amount"])
    return qp.density_matrix([0])
    

def encodeOne(row: dict):
    encodeCategorical(0, row["collateral_enc"], 4.0)
    encodeCategorical(1, row["purpose_enc"], 10.0)
    encodeCategorical(2, row["credit_history_enc"], 5.0)
    encodeCategorical(3, row["housing_enc"], 3.0)
    encodeNumerical(4, row["status_account_ord"], row["month_duration"])
    encodeNumerical(5, row["status_savings_ord"], row["years_employment_ord"])
    encodeNumerical(6, row["credit_amount"])
    return qp.density_matrix([0,1,2,3,4,5,6])
    

def encodeAll(dataset: pd.DataFrame, circuit) -> np.ndarray:
    
    density_matrices = []
    dict_df = dataset.to_dict(orient="records")
    for row in dict_df:
        rho = circuit(row)
        density_matrices.append(rho)
        
    return np.stack(density_matrices)

def encodeAllSingle(dataset: pd.DataFrame, circuit, i: int) -> np.ndarray:
    
    density_matrices = []
    dict_df = dataset.to_dict(orient="records")
    for row in dict_df:
        rho = circuit(row, i)
        density_matrices.append(rho)
        
    return np.stack(density_matrices)

def encodeExcludeOne(i: int, matrices: np.ndarray):
    return np.delete(matrices, i, axis=0)