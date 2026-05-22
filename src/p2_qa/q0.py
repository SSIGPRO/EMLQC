from math import *

import numpy as np

from matplotlib import pyplot as plt


from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram

def porta_sconosciuta(qc, qubit, tipo, angolo=0):
    """Simula la porta sconosciuta da identificare"""
    if tipo == 'ry':
        qc.ry(angolo, qubit)
    elif tipo == 'h':
        qc.h(qubit)
    # Si possono aggiungere altre porte

def circuito_identificazione(tipo_porta, angolo=0):
    qc = QuantumCircuit(2, 1)
    
    # Step 1: Preparazione
    qc.initialize([1, 0], 0)     # |0⟩ su q0
    qc.initialize([0, 1], 1)     # |1⟩ su q1
    
    # Step 2: Hadamard su entrambi
    qc.h(0)
    qc.h(1)  # q1 ora in |-⟩ = (|0⟩-|1⟩)/√2
    

    porta_sconosciuta(qc, 0, tipo_porta, angolo)
    qc.h(0)
    
    # Step 5: Misura q0
    qc.measure(0, 0)
    return qc

sampler = Sampler()
tipi_da_testare = [
    ('ry', 0),      # RY(0) = identità
    ('ry', np.pi),  # RY(π) = Pauli Y
    ('ry', np.pi/2),# RY(π/2)
    ('h', 0)        # Hadamard
]

for tipo, angolo in tipi_da_testare:
    qc = circuito_identificazione(tipo, angolo)
    
    # Mostra il circuito
    qc.draw('mpl')
    plt.show()

    job = sampler.run([qc], shots=1024)
    result = job.result()
    
    quasi = result.quasi_dists[0]
    counts = quasi.binary_probabilities()
    
    print(f"{tipo} (angolo={angolo}): {counts}")
    
    plot_histogram(counts)
    plt.show()
