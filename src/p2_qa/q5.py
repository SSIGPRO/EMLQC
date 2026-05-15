from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector

if __name__ == '__main__':
    sampler = Sampler()    
    shots = 1000
    phase = pi/4
    
    qc1 = QC(2)
    qc1.h(0)
    qc1.x(1)
    qc1.h(1)
    qc1.draw(output="mpl", interactive=True)
    qc1m = qc1.measure_all(inplace=False)

    qc2 = QC(2)
    qc2.h(0)
    qc2.x(1)
    qc2.h(1)
    qc2.cx(0,1)
    qc2.draw(output="mpl", interactive=True)
    qc2m = qc2.measure_all(inplace=False)

    job = sampler.run([qc1m, qc2m], shots=shots)
    results = job.result()
    for res, qc in zip(results, [qc1, qc2]):
        plot_bloch_multivector(Statevector(qc))
        plot_state_qsphere(qc)
    plt.show()

