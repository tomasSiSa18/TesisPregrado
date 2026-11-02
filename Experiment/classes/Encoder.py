import pandas as pd
import math
import pennylane as qp
import numpy as np

vars_params = {
    
    "collateral": [4], "purpose": [10], "credit_history": [5], "housing": [3], "other_installment_plans": [3], "status_account": [0, 3], "month_duration": [4, 72], "status_savings": [-0.5, 4.5], "years_employment": [-0.5,4.5], "credit_amount": [250, 18424]
    
}
    
def encodeCategorical(wire: int, value: float, k: float):
    qp.H(wires=wire)
    angle = (2*math.pi*value)/k
    qp.RZ(angle, wires=wire)

def encodeNumerical(wire: int, valueY: int, valueZ: int = 0):
    qp.RY(valueY, wires=wire)
    qp.RZ(valueZ, wires=wire)

def min_max(x: float, min: float, max: float):
    return math.pi * ((x-min)/(max-min))

def encodeOneColOneCable(row: dict, col: int, wire: int):
    if col == 0:
        encodeCategorical(wire, row["collateral"], vars_params["collateral"][0])
    elif col == 1:
        encodeCategorical(wire, row["purpose"], vars_params["purpose"][0])
    elif col == 2:
        encodeCategorical(wire, row["credit_history"], vars_params["credit_history"][0])
    elif col == 3:
        encodeCategorical(wire, row["housing"], vars_params["housing"][0])
    elif col == 4:
            encodeCategorical(wire, row["other_installment_plans"], vars_params["other_installment_plans"][0])
    elif col == 5:
        status_saving_norm = min_max(row["status_savings"], vars_params["status_savings"][0], vars_params["status_savings"][1])
        month_norm = min_max(row["month_duration"], vars_params["month_duration"][0], vars_params["month_duration"][1])
        assert(0 < status_saving_norm < math.pi)
        assert(0 <= month_norm <= math.pi)
        encodeNumerical(wire, status_saving_norm, month_norm)
    elif col == 6:
        years_employment_norm = min_max(row["years_employment"], vars_params["years_employment"][0], vars_params["years_employment"][1])
        credit_norm = min_max(row["credit_amount"], vars_params["credit_amount"][0], vars_params["credit_amount"][1])
        assert(0 < years_employment_norm < math.pi)
        assert(0 <= credit_norm <= math.pi)
        encodeNumerical(wire, years_employment_norm, credit_norm)
    elif col == 7:
        status_account_norm = min_max(row["status_account"], vars_params["status_account"][0], vars_params["status_account"][1])
        assert(0 <= status_account_norm <= math.pi)
        encodeNumerical(wire, status_account_norm)
    
def encodeOneCol(row: dict, col: int):

    encodeOneColOneCable(row, col, 0)
    return qp.density_matrix([0])

def encodeOneRow(row: dict):
   
    for i in range(0, 8):
        encodeOneColOneCable(row, i, i)
    
    return qp.density_matrix([0,1,2,3,4,5,6,7])
    

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