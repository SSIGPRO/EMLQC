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
    c1 = np.sqrt(np.array([0.7, 0.3]))

    qc = QC(3, 3)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)

    qc.cx(0,2)
    qc.cx(1,2)
    qc.ccx(0,1,2)

    qc.measure([0,1,2], [0,1,2])

    qc.draw(output="mpl", interactive=True)

    job = sampler.run([qc], shots=10000)
    result = job.result()
    print(result)
    print(result[0].data)
    counts = result[0].data['c'].get_counts()
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()
