from math import *
import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

def f00(qc): #Hadamard costant zero

    return 

def f01(qc):  #rotation ry
    qc.ry(pi/2, 1) 

    return

if __name__ == '__main__':
    sampler = Sampler()    
    
    qcs = []
    for foo in [f00, f01]:
        qc = QC(2, 1)
        qc.initialize([1.0, 0.0], 0)
        qc.initialize([0.0, 1.0], 1)

        qc.h(0)
        qc.h(1)
        foo(qc)
        qc.h(0)
        qc.h(1)

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

