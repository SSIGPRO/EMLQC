# general stuff
from math import *

# plot stuff
from matplotlib import pyplot as plt

# numpy
import numpy as np

# qiskit stuff
from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    qc = QC(2)
    c = np.array([1, 2, 3, 4])*(1/sqrt(30))
    qc.initialize(c)
    qc.draw(output="mpl", interactive=True)
    #plt.show()

    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data['meas'].get_counts()}")
    counts = result[0].data['meas'].get_counts()
    plot_histogram(counts)
    plt.show()
