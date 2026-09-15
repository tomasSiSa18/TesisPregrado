import pennylane as qp

device = qp.device('default.mixed', wires=2)

@qp.qnode(device)
def bitflip(x):
    qp.X(wires=x)
    return qp.density_matrix([0,1])

print(bitflip(0))