from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector

def Ry(qc):
    theta = pi/2;
    qc.ry(theta, 0)
    qc.cx(0,1)
    qc.ry(theta,0)
    return

def H(qc):
    qc.h(0)
    qc.cx(0,1)
    qc.h(0)

    return

if __name__ == '__main__':
    sampler = Sampler()    
    shots = 10
    qc = QC(2)
    
    qc.initialize([1.0, 0.0], 0)
    qc.initialize([0.0, 1.0], 1)

    qc.h(0)
    qc.h(1)

    qc.barrier()

    #H function for hadamard gate and Ry for rotatio gate
    
    #H(qc)
    #Ry(qc)

    qc.barrier()

    qc.h(0)
    qc.h(1)

    qc.draw(output="mpl", interactive=True)
    qcm = qc.measure_all(inplace=False)

    job = sampler.run([qcm], shots=shots)
    results = job.result()
    counts = results[0].data['meas'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})
    print(f" > Counts: {counts}")
    plot_bloch_multivector(Statevector(qc))
    plt.show()

