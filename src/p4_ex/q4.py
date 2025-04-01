from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC, QuantumRegister
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector

from qiskit.circuit.library import GroverOperator, MCMTGate, ZGate
from qiskit.circuit.library import DraperQFTAdder as Add

if __name__ == '__main__':
    shots = 1000
    sampler = Sampler()    
    
    # circuit
    qc = QC(3)

    # setting input
    qc.x(0)
    #qc.x(1)

    qc.barrier()
    qc.ccx(0, 1, 2)
    qc.x(2)

    qc.barrier()

    
    qc.draw(output="mpl", interactive=True)
    qcm = qc.measure_all(inplace=False)

    
    job = sampler.run([qcm], shots=shots)
    results = job.result()
    counts = results[0].data['meas'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})
    print(f" > Counts: {counts}")
    plt.show()

