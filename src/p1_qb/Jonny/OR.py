from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    sampler = Sampler()    
    
    # Define the initial states for the qubits
    # In order to create an OR gate, we have to set the ancilla bit to |0>
    ca = np.sqrt(np.array([1.0, 0.0]))
    c1 = np.sqrt(np.array([0.0, 1.0]))
    c2 = np.sqrt(np.array([0.0, 1.0]))
    
    qc = QC(3)
    qc.initialize(ca, 0) # Ancilla bit
    qc.initialize(c1, 1)
    qc.initialize(c2, 2)

    qc.cx([1, 2], 0) #thanks to these two CNOTs, the ancilla bit is flipped to |1> if either of the input bits is |1>
    qc.ccx(1, 2, 0) #thanks to this Toffoli gate, the ancilla bit is flipped to |1> if both input bits are |1>

    qc.draw(output="mpl", interactive=True)
    
    qcm = qc.measure_all(inplace=False)
    job = sampler.run([qcm], shots=1000)
    result = job.result()
    counts = result[0].data['meas'].get_counts()
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()

