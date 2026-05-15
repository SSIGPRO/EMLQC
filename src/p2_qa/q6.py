from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector

def get_qc(theta):
    qc = QC(2, 1)
    qc.x(1)
    qc.barrier()
                             
    qc.h(0)
    qc.barrier()
    
    qc.cp(2*pi*theta, 0, 1)
    
    qc.barrier()
    qc.h(0)
    return qc

if __name__ == '__main__':
    sampler = Sampler()    
    shots = 1000
    theta = 1/6
   
    qc1 = get_qc(theta)
    qc2 = get_qc(theta)
    qc2.measure(qubit=0, cbit=0)

    qc2.draw(output="mpl", interactive=True)

    job = sampler.run([qc2], shots=shots)
    results = job.result()
    counts = results[0].data['c'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})

    plot_bloch_multivector(Statevector(qc1))
    plot_state_qsphere(qc1)
    plt.show()

