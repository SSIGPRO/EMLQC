from math import *

import numpy as np
import torch
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

if __name__ == "__main__":
    sampler = Sampler()
    c0 = np.sqrt(np.array([0.3, 0.7]))
    c1 = np.sqrt(np.array([0.8, 0.2]))

    qc = QC(2, 2)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)
    qc_measured = qc.measure_all(inplace=False, add_bits=False)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=0, cbit=0)
    qc.draw(output="mpl", interactive=True)
    job_before = sampler.run([qc_measured], shots=1000)
    result_before = job_before.result()
    counts_before = result_before[0].data["c"].get_counts()
    plot_histogram(counts_before)
    job = sampler.run([qc], shots=1000)
    result = job.result()
    print(result)
    print(result[0].data)
    counts = result[0].data["c"].get_counts()
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()
