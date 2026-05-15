from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector

if __name__ == '__main__':
    sampler = Sampler()    
    shots = 1000

    qc_not = QC(2)
    qc_not.x(0)
    qc_not.h(1)
    qc_not.cz(0, 1)
    qc_not.h(1)
    
    qc_not.draw(output="mpl", interactive=True)
    qcm = qc_not.measure_all(inplace=False)

    job = sampler.run([qcm], shots=shots)
    results = job.result()
    counts = results[0].data['meas'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})
    print(f" > Counts: {counts}")
    plot_bloch_multivector(Statevector(qc_not))
    plot_state_qsphere(qc_not)
    
    qc_id = QC(2)
    qc_id.h(1)
    qc_id.cz(0, 1)
    qc_id.h(1)
    
    qc_id.draw(output="mpl", interactive=True)
    qcm = qc_id.measure_all(inplace=False)

    job = sampler.run([qcm], shots=shots)
    results = job.result()
    counts = results[0].data['meas'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})
    print(f" > Counts: {counts}")
    plot_bloch_multivector(Statevector(qc_id))
    plot_state_qsphere(qc_id)
    plt.show()

