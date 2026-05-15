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

    qc = QC(2)
    qc.h(0)
    qc.draw(output="mpl", interactive=True)
    qcm = qc.measure_all(inplace=False)

    job = sampler.run([qcm], shots=shots)
    results = job.result()
    counts = results[0].data['meas'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})
    print(f" > Counts: {counts}")
    plot_bloch_multivector(Statevector(qc))
    plot_state_qsphere(qc)
    plt.show()

