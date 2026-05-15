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

    c0 = np.sqrt(np.array([1.0, 0.0]))
    c1 = np.sqrt(np.array([1.0, 0.0]))

    #qubit a 0
    qc0 = QC(2)
    qc0.initialize(c0, 0)
    qc0.initialize(c1, 1)
    qc0.h(1)
    qc0.cz(0,1)
    qc0.h(1)
    qc0.draw(output="mpl", interactive=True)
    qcm0 = qc0.measure_all(inplace=False)

    #qubit a 1
    qc1 = QC(2)
    qc1.initialize(c0, 0)
    qc1.initialize(c1, 1)
    qc1.x(0)
    qc1.h(1)
    qc1.cz(0,1)
    qc1.h(1)
    qc1.draw(output="mpl", interactive=True)
    qcm1 = qc1.measure_all(inplace=False)


    job = sampler.run([qcm0,qcm1], shots=shots)
    results = job.result()
    counts0 = results[0].data['meas'].get_counts()
    counts1 = results[1].data['meas'].get_counts()
    plot_histogram({k:c/shots for k, c in counts0.items()})
    plot_histogram({k:c/shots for k, c in counts1.items()})
    print(f" > Counts: {counts0}")
    print(f" > Counts: {counts1}")
    plot_bloch_multivector(Statevector(qc0))
    plot_state_qsphere(qc0)
    plot_bloch_multivector(Statevector(qc1))
    plot_state_qsphere(qc1)
    plt.show()

