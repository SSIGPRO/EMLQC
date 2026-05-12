from math import *

import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info.operators import Operator

if __name__ == '__main__':
    sampler = StatevectorSampler()    
    coeffs = np.array([1/sqrt(10),3/sqrt(10)]) 
    inv = np.array([[0, 1], [1, 0]])

    qc = QC(1)
    qc.initialize(coeffs, 0)
    qc.x(0)
    qc.draw(output="mpl", interactive=True)
    qc_measured = qc.measure_all(inplace=False)

    qc2 = QC(1)
    qc2.initialize(inv@coeffs, 0)
    qc2.draw(output="mpl", interactive=True)
    qc2_measured = qc2.measure_all(inplace=False)

    qc3 = QC(1)
    inv_op = Operator(inv)
    qc3.initialize(coeffs, 0)
    qc3.unitary(inv_op, [0], 'why not?')
    qc3.draw(output="mpl", interactive=True)
    qc3_measured = qc3.measure_all(inplace=False)

    job = sampler.run([qc_measured, qc2_measured, qc3_measured], shots=1000)
    result = job.result()

    print(f" > Counts: {result[0].data['meas'].get_counts()}")
    print(f" > Counts: {result[1].data['meas'].get_counts()}")
    print(f" > Counts: {result[2].data['meas'].get_counts()}")
    plt.show()

