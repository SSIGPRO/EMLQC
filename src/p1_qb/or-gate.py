from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':

    sampler = Sampler()
    num_samples = 10

    # Caso: 11
    c0 = np.sqrt(np.array([0, 1])) 
    c1 = np.sqrt(np.array([0, 1])) 

    qc = QC(3, 3)

    qc.initialize(c0, 0)
    qc.initialize(c1, 1)

    qc.cx(0, 2)     # se q0 = 1 -> q2 viene invertito
    qc.cx(1, 2)     # se q1 = 1 -> q2 viene invertito
    qc.ccx(0,1,2)   # se q0 = 1 e q1 = 1 -> q2 viene invertito    

    qc.measure(qubit=0, cbit=0)
    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=2, cbit=2)
    qc.draw(output="mpl")
    plt.show()

    job = sampler.run([qc], shots = num_samples)
    result = job.result()
    print(result)
    print(result[0].data)
    counts = result[0].data['c'].get_counts()
    print(f" > Counts: {counts}")
    plot_histogram(counts)
    plt.show()