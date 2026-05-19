from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    sampler = Sampler()    
    c0 = np.sqrt(np.array([0.0, 1.0]))
    c1 = np.sqrt(np.array([0.0, 1.0]))
    c2 = np.sqrt(np.array([1.0, 0.0]))

    qc0 = QC(3)
    qc0.initialize(c0, 0)
    qc0.initialize(c1, 1)
    qc0.initialize(c2, 2)
    qc0.h([0, 1, 2])
    qc0.ccx(0, 1, 2)
    qc0.h([0, 1, 2])
    qc0.draw(output="mpl", interactive=True)

    qc1 = QC(3)
    qc1.initialize(c0, 0)
    qc1.initialize(c1, 1)
    qc1.initialize(c2, 2)
    
    qc0_measured = qc0.measure_all(inplace=False)
    qc1_measured = qc1.measure_all(inplace=False)
    job = sampler.run([qc0_measured, qc1_measured], shots=1000)
    result = job.result()
    for res in result:
        counts = res.data['meas'].get_counts()
        plot_histogram(counts)
        print(f" > Counts: {counts}")
    plt.show()

