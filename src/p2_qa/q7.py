import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.basis_change.qft import QFTGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit.visualization import (plot_bloch_multivector, plot_histogram, plot_state_qsphere)


def f(theta: float) -> Gate:
    qc = QuantumCircuit(1, name="F")
    qc.h(0)
    qc.ry(theta, 0)
    qc.rz(theta, 0)
    
    return qc.to_gate().control(1)


def get_qc(theta):
    qc = QC(3, 2)
    
    qc.x(2)
    qc.barrier()

    qc.h(0)
    qc.h(1)
    qc.barrier()

    gate = f(theta)
    
    qc.append(gate, [0, 2])
    qc.barrier()

    qc.append(gate, [1, 2])
    qc.append(gate, [1, 2])
    qc.barrier()

    qc.compose(QFTGate(2).inverse(), qubits=[0, 1], inplace=True)
    return qc

if __name__ == "__main__":
    sampler = Sampler()
    shots = 1000
    
    theta = np.pi / 2

    qc1 = get_qc(theta)
    qc2 = get_qc(theta)
    
    qc2.measure(qubit=0, cbit=0)
    qc2.measure(qubit=1, cbit=1)

    qc2.draw(output="mpl", interactive=True)

    job = sampler.run([qc2], shots=shots)
    results = job.result()
    counts = results[0].data["c"].get_counts()
    plot_histogram({k: c / shots for k, c in counts.items()})

    plot_bloch_multivector(Statevector(qc1))
    plot_state_qsphere(qc1)
    plt.show()