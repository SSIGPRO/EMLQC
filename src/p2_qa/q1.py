from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

def f00(qc):
    qc.ry(pi/4, 0)
    qc.cx(0,1)
    qc.ry(pi/4, 0)
    return

def f01(qc):
    return

if __name__ == '__main__':
    sampler = Sampler()    
    shots = 10000

    qcs = []
    for foo in [f00, f01]:
        qc = QC(2)
        qc.initialize([1.0, 0.0], 0)
        qc.initialize([0.0, 1.0], 1)
        qc.h(0)
        qc.h(1)
        foo(qc)
        qc.h(0)
        qc.h(1)
        qc.draw(output="mpl", interactive=True)
        qcm = qc.measure_all(inplace=False)
        qcs.append(qcm)

    job = sampler.run(qcs, shots=shots)
    results = job.result()
    for res in results:
        counts = res.data['meas'].get_counts()
        plot_histogram({k:c/shots for k, c in counts.items()})
        print(f" > Counts: {counts}")
    plt.show()

