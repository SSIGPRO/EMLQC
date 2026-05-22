from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT


#funzione per avere il gate controllato
def F_to_controlled_gate_F(theta, alpha, qc_finale, qubit_controller, qubit_target):

    #faccio il circuito per la mia funzione custom da controllare
    qc_F = QC(1, name="F")
    qc_F.ry(alpha, 0)
    qc_F.rz(2*np.pi*theta, 0)

    #converto il circuito in un gate
    gate_F = qc_F.to_gate()

    #implemento il controllo del gate
    controlled_gate_F = gate_F.control(1)           #oppure anche qc_F.to_gate().control(1)
    qc_finale.append(controlled_gate_F, [qubit_controller, qubit_target])
    return qc_finale


def get_qc(theta, alpha):
    qc = QC(3, 2)
    #di default sono inizializzati tutti e 3 i qubit a zero

    qc.x(2)
    qc.barrier()
                             
    qc.h(0)
    qc.h(1)
    qc.barrier()
    
    F_to_controlled_gate_F(theta, alpha, qc, 0, 2)

    qc.barrier()

    F_to_controlled_gate_F(theta, alpha, qc, 1, 2)
    F_to_controlled_gate_F(theta, alpha, qc, 1, 2)

    qc.barrier()
    qc.compose(
        QFT(2, inverse=True).decompose(),
        inplace=True
    )
    return qc


if __name__ == '__main__':
    sampler = Sampler()    
    shots = 1000
    theta = 3/4
    alpha = np.pi/6
   
    qc1 = get_qc(theta, alpha)

    qc2 = get_qc(theta, alpha)
    qc2.measure(qubit=0, cbit=0)
    qc2.measure(qubit=1, cbit=1)

    qc2.draw(output="mpl", interactive=True)

    job = sampler.run([qc2], shots=shots)
    results = job.result()
    counts = results[0].data['c'].get_counts()
    plot_histogram({k:c/shots for k, c in counts.items()})
    print(f" > Counts: {counts}")
    plot_bloch_multivector(Statevector(qc1))
    plot_state_qsphere(qc1)
    plt.show()