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
    c1 = np.sqrt(np.array([1.0, 0.0]))
    ca = np.sqrt(np.array([0.0, 1.0]))
    
    qc = QC(3)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)
    qc.initialize(ca, 2)
    
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    
    qc.draw(output="mpl", interactive=True)
    
    qcm = qc.measure_all(inplace=False)
    job = sampler.run([qcm], shots=1000)
    result = job.result()
    counts = result[0].data['meas'].get_counts()
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()

