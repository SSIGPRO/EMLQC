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

    #incremento della probabilità del 10% (+0.1)
    #Si usa la radice quadrata poiché la probabilità è il quadrato dell'ampiezza indotta dal seno
    alpha = -theta + 2*asin(sqrt(0.1+(sin(theta/2))**2))

# Primo Circuito (Stato Iniziale)
    qc = QC(1)
    qc.initialize(coeffs, 0)
    
    # Rimosso interactive=True per evitare incompatibilità grafiche
    qc.draw(output="mpl")
    qc_measured = qc.measure_all(inplace=False)

 # Secondo Circuito (Stato Ruotato)
    qc2 = QC(1)
    qc2.initialize(coeffs, 0)
    qc2.ry(alpha, 0)
    qc2.draw(output="mpl")
    qc2_measured = qc2.measure_all(inplace=False)

# Esecuzione c
    job = sampler.run([qc_measured, qc2_measured], shots=1000)
    result = job.result()

    print(f" > Angolo calcolato alpha: {alpha:.4f} radianti")
    print(f" > Counts: {result[0].data['meas'].get_counts()}")
    print(f" > Counts: {result[1].data['meas'].get_counts()}")
    plt.show()