from math import *
import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    sampler = Sampler()    
    
    c0 = np.sqrt(np.array([0.1, 0.9]))
    c1 = np.sqrt(np.array([0.5, 0.5]))

    qc = QC(2, 2)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)
    
    qc.cx(1, 0)
    
    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=0, cbit=0)

    job = sampler.run([qc], shots=1)
    result = job.result()
    
    counts = result[0].data['c'].get_counts()
    print(f" > Risultato misurazione (collasso): {counts}")
    
    plot_histogram(counts)
    plt.show()