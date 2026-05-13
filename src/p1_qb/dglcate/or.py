from math import *

import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    sampler = Sampler()    
    c0 = np.sqrt(np.array([1.0, 0.0]))
    c1 = np.sqrt(np.array([0.0, 1.0]))  #come esempio proviamo l'or tra 0 (c0) e 1 (c1)
    c2 = np.sqrt(np.array([1.0, 0.0]))  #come risultato devo ottenere 1, messo in c2


    #implementazione dell'or
    qc = QC(3,3)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)
    qc.initialize(c2, 2)
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=0, cbit=0)
    qc.measure(qubit=2, cbit=2)
    qc.draw(output="mpl", interactive=True)

    job = sampler.run([qc], shots=1000)
    result = job.result()
    print(result)
    print(result[0].data)
    counts = result[0].data['c'].get_counts()
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()