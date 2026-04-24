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

    qc_before = qc.measure_all(inplace=False, add_bits=False)

    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)

    job_before = sampler.run([qc_before], shots=1000)
    result_before = job_before.result()
    counts_before = result_before[0].data["c"].get_counts()
    print(f"counts before swap: {counts_before}")
    plot_histogram(counts_before, title="before swap")

    qc.draw(output="mpl", interactive=True)

    job = sampler.run([qc], shots=1000)
    result = job.result()
    counts = result[0].data["c"].get_counts()
    print(f"counts after swap: {counts}")
    plot_histogram(counts, title="after swap")

    plt.show()