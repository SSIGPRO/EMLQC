from math import *

import numpy as np
import torch
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram


def f0(qc: QuantumCircuit):
    qc.h(1)
    return


def f1(qc: QuantumCircuit):
    theta = np.pi / 6
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.ry(theta, 0)
    return


if __name__ == "__main__":
    sampler = Sampler()

    qcs = []
    for foo in [f0, f1]:
        qc = QC(2, 1)
        qc.initialize([1.0, 0.0], 0)
        qc.initialize([0.0, 1.0], 1)
        qc.h(0)
        qc.h(1)
        foo(qc)
        qc.h(0)
        qc.measure(qubit=0, cbit=0)
        qc.draw(output="mpl", interactive=True)
        qcs.append(qc)

    job = sampler.run(qcs, shots=1000)
    results = job.result()
    for res in results:
        counts = res.data["c"].get_counts()
        plot_histogram(counts)
        print(f" > Counts: {counts}")
    plt.show()
