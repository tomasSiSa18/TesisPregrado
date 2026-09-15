from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
import math

backend = FakeSherbrooke()
qp = backend.qubit_properties(0)
t1 = qp.t1

gate_time = backend.target.durations().get("sx", 0, unit="s")
gamma = 1 - math.exp(-gate_time / t1)
print(gamma)