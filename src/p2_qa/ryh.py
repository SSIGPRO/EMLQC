from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

def f_Ry(qc):
    qc.ry(pi/4, 0)
    qc.cx(0, 1)     
    qc.ry(pi/4, 0)
    
def f_h(qc):
    qc.h(0)
    qc.cx(0, 1)      
    qc.h(0)

if __name__ == '__main__':
    sampler = Sampler()    
    
    qcs = []
    for foo in [f_Ry, f_h]:
        qc = QC(2, 1)
        qc.initialize([1.0, 0.0], 0)
        qc.initialize([0.0, 1.0], 1)
        qc.h(0)
        qc.h(1)
        foo(qc)
        qc.h(0)
        qc.measure(qubit=0,cbit=0)
        qc.draw(output="mpl", interactive=True)
        qcs.append(qc)

    job = sampler.run(qcs, shots=1000)
    results = job.result()
    for res in results:
        counts = res.data['c'].get_counts()
        plot_histogram(counts)
        print(f" > Counts: {counts}")
    plt.show()

