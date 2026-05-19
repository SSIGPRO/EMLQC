from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

def f00(qc): #costant zero

    return 

def f01(qc): #indentity 
    qc.cx(0, 1) 

    return

def f10(qc): # c-not 
    qc.cx(0,1)
    qc.x(1)
    
    return

def f11(qc): #costant one 
    qc.x(1)
    return

if __name__ == '__main__':
    sampler = Sampler()    
    
    qcs = []
    for foo in [f00, f01, f10, f11]:
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

