from math import *

import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    sampler = StatevectorSampler()    
    coeffs = np.sqrt(np.array([0.1, 0.9]))
    coeffs2 = np.array([1/sqrt(2), 1/sqrt(2)]) 

    qc = QC(2)
    qc.initialize(coeffs, 0)
    qc.initialize(coeffs2, 1)
    qc.cx(1, 0) #qbit 1 controls qbit 0

    qc.draw(output="mpl", interactive=True)
    qc_measured = qc.measure_all(inplace=False)

    qc2 = QC(2)
    qc2.initialize(np.sqrt([0.05, 0.45, 0.45, 0.05]))
    qc2.draw(output="mpl", interactive=True)
    qc2_measured = qc2.measure_all(inplace=False)

    job = sampler.run([qc_measured, qc2_measured], shots=1000)
    result = job.result()
    counts = result[0].data["meas"].get_counts()
    counts2 = result[1].data["meas"].get_counts()
    plot_histogram(counts)
    plot_histogram(counts2)
    print(f" > Counts: {counts}")
    plt.show()

