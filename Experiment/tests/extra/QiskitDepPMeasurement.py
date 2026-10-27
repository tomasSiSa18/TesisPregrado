from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibm_marrakesh")

props = backend.properties()

gate_error = props.gate_error("sx", [0])

def infidelity_to_depolarizing_p(avg_infidelity, num_qubits):
    d = 2 ** num_qubits
    return avg_infidelity * d / (d - 1)

p_1q = infidelity_to_depolarizing_p(gate_error, 1)

print(p_1q)