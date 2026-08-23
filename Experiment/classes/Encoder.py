import pandas as pd
import math
import pennylane as qp

class Encoder:
    
    def __init__(self, device, dataset: pd.DataFrame):
        self.device = device
        self.dataset = dataset
        #self.categorical_rows = categorical_rows
        #self.numerical_rows = numerical_rows
        
        self.circuit = qp.QNode(self.encodeOne, self.device)
    
    def encodeCategorical(self, wire: int, value: float, k: float):
        qp.H(wires=wire)
        qp.RZ((2*math.pi*value)/k, wires=wire)
    
    def encodeNumerical(self, wire: int, valueY: int, valueZ: int = 0):
        
        qp.RY(valueY, wires=wire)
        qp.RZ(valueZ, wires=wire)
    
    def encodeOne(self, row: dict):
        self.encodeCategorical(0, row["collateral_enc"], 4.0)
        self.encodeCategorical(1, row["purpose_enc"], 10.0)
        self.encodeCategorical(2, row["credit_history_enc"], 5.0)
        self.encodeCategorical(3, row["housing_enc"], 3.0)
        self.encodeNumerical(4, row["status_account_ord"], row["month_duration"])
        self.encodeNumerical(5, row["status_savings_ord"], row["years_employment_ord"])
        self.encodeNumerical(6, row["credit_amount"])
        return qp.density_matrix([0,1,2,3,4,5,6])
        
    
    def encodeMany(self):
        
        dict_df = self.dataset.to_dict(orient="records")
        for row in dict_df:
            self.encodeOne(row)
            