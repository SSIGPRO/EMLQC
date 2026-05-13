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
    c0 = np.sqrt(np.array([0.1, 0.9]))
    c1 = np.sqrt(np.array([0.0, 1.0]))


    #qubit normale circuito AB 
    qc = QC(2, 2)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)
    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=0, cbit=0)
    qc.draw(output="mpl", interactive=True)

    #qubit swap circuito BA + 3cnot
    qc_swap = QC(2, 2)
    qc_swap.initialize(c1, 0)
    qc_swap.initialize(c0, 1)
    qc_swap.cx(0,1)
    qc_swap.cx(1,0)
    qc_swap.cx(0,1)
    qc_swap.measure(qubit=1, cbit=1)
    qc_swap.measure(qubit=0, cbit=0)
    qc_swap.draw(output="mpl", interactive=True)
    

    job = sampler.run([qc, qc_swap], shots=1000)
    result = job.result()
    print(result)
    print(result[0].data)
    counts = result[0].data['c'].get_counts()
    counts_cnot = result[1].data['c'].get_counts()
    plot_histogram(counts)
    plot_histogram(counts_cnot)
    print(f" > Counts: {counts}")
    print(f" > Counts: {counts_cnot}")
    plt.show()

