from math import *

import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':
    sampler = StatevectorSampler()    
    theta = (pi/4)
    coeffs = np.array([cos(theta/2), sin(theta/2)])
    alpha = -theta + 2*asin(sqrt(0.1+(sin(theta/2))**2))

    qc = QC(1)
    qc.initialize(coeffs, 0)
    qc.draw(output="mpl", interactive=True)
    qc_measured = qc.measure_all(inplace=False)

    qc2 = QC(1)
    qc2.initialize(coeffs, 0)
    qc2.ry(alpha, 0)
    qc2.draw(output="mpl", interactive=True)
    qc2_measured = qc2.measure_all(inplace=False)

    job = sampler.run([qc_measured, qc2_measured], shots=1000)
    result = job.result()

    print(f" > Counts: {result[0].data['meas'].get_counts()}")
    print(f" > Counts: {result[1].data['meas'].get_counts()}")
    plt.show()