from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT

def F_gate(theta):
    qc = QC(1, name = "F")

    qc.rz(theta, 0)
    return qc.to_gate().control(1)


def get_qc(theta):
    qc = QC(3, 2)
    qc.x(2)

    qc.barrier()

    qc.h(0)
    qc.h(1)

    qc.barrier()

    qc.append(F_gate(theta), [0, 2])

    qc.barrier()

    qc.append(F_gate(theta), [1, 2])
    qc.append(F_gate(theta), [1, 2]) 

    qc.barrier()

    qc.compose(
        QFT(2, inverse=True).decompose(),
        inplace=True
    )
    return qc
   
if __name__ == '__main__':
    sampler = Sampler()    
    shots = 1000
    theta = pi
   
    qc1 = get_qc(theta)

    qc2 = get_qc(theta)

    qc2.measure(qubit=0, cbit=0)
    qc2.measure(qubit=1, cbit=1)

    qc2.draw(output="mpl", interactive=True)

    job = sampler.run([qc2], shots=shots)
    results = job.result()
    counts = results[0].data['c'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})

    #plot_bloch_multivector(Statevector(qc1))
    #plot_state_qsphere(qc1)

    plt.show()
