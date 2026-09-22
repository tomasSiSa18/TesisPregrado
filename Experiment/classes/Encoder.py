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

def min_max(x: float, min: float, max: float):
    return (x-min)/max-min

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
        
        status_account_norm = min_max(row["status_account_ord"], 0, 3)
        month_norm = min_max(row["month_duration"], 4, 72)
        
        encodeNumerical(0, status_account_norm, month_norm)
    elif row_num == 5:
        status_saving_norm = min_max(row["status_savings_ord"], 1, 5)
        years_employment_norm = min_max(row["years_employment_ord"], 1, 5)
        encodeNumerical(0, status_saving_norm, years_employment_norm)
    elif row_num == 6:
        credit_norm = min_max(row["credit_amount"], 250, 18400)
        encodeNumerical(0, credit_norm)
    return qp.density_matrix([0])
    

def encodeOne(row: dict):
    encodeCategorical(0, row["collateral_enc"], 4.0)
    encodeCategorical(1, row["purpose_enc"], 10.0)
    encodeCategorical(2, row["credit_history_enc"], 5.0)
    encodeCategorical(3, row["housing_enc"], 3.0)
    status_account_norm = min_max(row["status_account_ord"], 0, 3)
    month_norm = min_max(row["month_duration"], 4, 72)
    
    encodeNumerical(4, status_account_norm, month_norm)
    status_saving_norm = min_max(row["status_savings_ord"], 1, 5)
    years_employment_norm = min_max(row["years_employment_ord"], 1, 5)
    encodeNumerical(5, status_saving_norm, years_employment_norm)
    credit_norm = min_max(row["credit_amount"], 250, 18400)
    encodeNumerical(6, credit_norm)
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