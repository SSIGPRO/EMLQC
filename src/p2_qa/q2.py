from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

from qutip import Bloch

def plt_bloch(theta, phi):
    b = Bloch()
    b.clear()
    v = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
    b.add_vectors(v)
    return b

if __name__ == '__main__':
    sampler = Sampler()    
    shots = 1000
    theta = pi/2
    phi = 0

    qc = QC(1)
    qc.initialize([cos(theta/2), complex(cos(phi), sin(phi))*sin(theta/2)], 0)
    b = plt_bloch(theta, phi)
    qc.draw(output="mpl", interactive=True)
    qcm = qc.measure_all(inplace=False)

    job = sampler.run([qcm], shots=shots)
    results = job.result()
    counts = results[0].data['meas'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})
    print(f" > Counts: {counts}")
    b.show()
    plt.show()

