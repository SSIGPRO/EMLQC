from math import *

import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    sampler = Sampler()    
    c0 = np.sqrt(np.array([0.1, 0.9]))
    c1 = np.sqrt(np.array([0.8, 0.2]))

    #Qubit normale
    qc = QC(2, 2)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)

    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=0, cbit=0)
    qc.draw(output="mpl", interactive=True)

    #Qubit swapped
    qc_swapped = QC(2, 2)
    qc_swapped.initialize(c0, 0)
    qc_swapped.initialize(c1, 1)

    qc_swapped.cx(1,0)
    qc_swapped.cx(0,1)
    qc_swapped.cx(1,0)

    qc_swapped.measure(qubit=1, cbit=1)
    qc_swapped.measure(qubit=0, cbit=0)
    qc_swapped.draw(output="mpl", interactive=True)

    job = sampler.run([qc, qc_swapped], shots=1000)
    result = job.result()
    print(result)
    print(result[0].data)
    counts = result[0].data['c'].get_counts()
    counts_swapped = result[1].data['c'].get_counts()
    plot_histogram([counts, counts_swapped], legend=['Normale', 'Swapped'])
    print(f" > Counts normale: {counts}")
    print(f" > Counts swapped: {counts_swapped}")

    plt.show()

