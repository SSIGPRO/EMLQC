from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC, QuantumRegister
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector

from qiskit.circuit.library import GroverOperator, MCMTGate, ZGate, QuadraticFormGate
from qiskit.circuit.library import DraperQFTAdder as Add

def n_iters(nqb):
    ni = pi / (4 * asin(sqrt(1 / 2**nqb)))
    #ni = (pi/4)*sqrt(2**nqb)
    return floor(ni)

def get_oracle():
    b = QuantumRegister(1, 'b')
    ris = QuantumRegister(2, 'ris')

    qc = QC(*[b, ris])

    # First H from marking 
    qc.h(ris)
    qc.barrier()

    quadratic_matrix = [[1]]
    linear_vector = [-2]
    qf_gate = QuadraticFormGate(num_result_qubits=2, quadratic=quadratic_matrix, linear=linear_vector)
    
    qc.append(qf_gate, b[:]+ris[:])
    qc.barrier()

    qc.h(ris[1])
    qc.compose(MCMTGate(ZGate(), 1, 1), ris[:], inplace=True)
    qc.x(ris)
    qc.barrier()

    # uncompute f
    qc.append(qf_gate.inverse(), b[:]+ris[:])
    
    # Last H from the marking
    qc.h(ris)
    qc.barrier()

    return qc


if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    print('niter: ', ni)
    shots = 1000
    sampler = Sampler()    
    
    # Grover's stuff
    oracle = get_oracle() 
    grover_op = GroverOperator(oracle, reflection_qubits = [1, 2])
    oracle.draw(output="mpl", interactive=True)

    # circuit
    qc = QC(3, 2)
    
    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()
    qc.measure([1, 2], [0, 1])

    qc.draw(output="mpl", interactive=True)
    
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = {k:v/shots for k, v in result[0].data['c'].get_counts().items()}
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()