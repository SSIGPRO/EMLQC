from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector

def H(qc):
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    return

def Ry(qc):
    alpha = np.pi/2
    qc.ry(alpha, 0)
    qc.cx(0, 1)
    qc.ry(alpha, 0)
    return


if __name__ == '__main__':
    sampler = Sampler()    
    
    qcs = []
    for function in [H, Ry]:
        qc = QC(2, 1)
        qc.initialize([1.0, 0.0], 0)
        qc.initialize([0.0, 1.0], 1)
        qc.h(0)
        qc.h(1)
        function(qc)
        qc.h(0)
        plot_bloch_multivector(Statevector(qc))
        plot_state_qsphere(qc)
        qc.measure(qubit=0,cbit=0)
        qc.draw(output="mpl", interactive=True)
        qcs.append(qc)

    job = sampler.run(qcs, shots=1000)
    results = job.result()
    for res in results:
        counts = res.data['c'].get_counts()
        plot_histogram(counts)
        print(f" > Counts: {counts}")
    plt.show()