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
    
    inp_a = 1 
    inp_b = 0
    
    qc = QC(3, 1)

    if inp_a == 1:
        qc.x(0)
    if inp_b == 1:
        qc.x(1)

    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)

    qc.measure(2, 0)

    qc.draw(output="mpl")
    plt.show()

    job = sampler.run([qc], shots=10000)
    result = job.result()
    
    counts = result[0].data.c.get_counts()
    
    print(f"Inputs: A={inp_a}, B={inp_b}")
    print(f" > Counts: {counts}")
    
    plot_histogram(counts)
    plt.show()