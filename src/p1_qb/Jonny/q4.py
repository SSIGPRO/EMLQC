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
    c0 = np.sqrt(np.array([0.6, 0.4]))
    c1 = np.sqrt(np.array([0.3, 0.7]))


    qc = QC(2, 2)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)
    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=0, cbit=0)
    #qc_measured = qc.measure_all(inplace=False)
    qc.draw(output="mpl", interactive=True)

    qc_cnot = QC(2, 2)
    qc_cnot.initialize(c0, 0)
    qc_cnot.initialize(c1, 1)
    qc_cnot.cx(1,0)
    qc_cnot.measure(qubit=1, cbit=1)
    qc_cnot.measure(qubit=0, cbit=0)
    #qc_cnot_measured = qc_cnot.measure_all(inplace=False)
    qc_cnot.draw(output="mpl", interactive=True)

    qc_swap = QC(2, 2)
    qc_swap.initialize(c1, 0)
    qc_swap.initialize(c0, 1)
    qc_swap.cx(1,0)
    qc_swap.cx(0,1)
    qc_swap.cx(1,0)
    #qc_swap_measured = qc_swap.measure_all(inplace=False)
    qc_swap.measure(qubit=1, cbit=1)
    qc_swap.measure(qubit=0, cbit=0)
    qc_swap.draw(output="mpl", interactive=True)

    job = sampler.run([qc, qc_cnot, qc_swap], shots=1000)
    result = job.result()
    print(result)
    counts = result[0].data['c'].get_counts()
    counts2 = result[1].data['c'].get_counts()
    counts3 = result[2].data['c'].get_counts()
    plot_histogram(counts)
    plot_histogram(counts2)
    plot_histogram(counts3)
    print(f" > Counts: {counts}")
    print(f" > Counts: {counts2}")
    print(f" > Counts: {counts3}")
    plt.show()

