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
from qiskit.circuit.library import QuadraticFormGate as QF

def n_iters(nqb, nsol):
    ni = pi / (4 * asin(sqrt(nsol / 2**nqb)))
    #ni = (pi/4)*sqrt(2**nqb)
    return floor(ni)

def get_oracle():
    x = QuantumRegister(3, 'x')
    sol = QuantumRegister(4, 'sol')

    qc = QC(x, sol)

    quad = QF(4, [[0,4,8],[0,0,0],[0,0,0]], [15,0,8], 1)

    qc.append(quad, x[:] + sol[:])
    qc.barrier() 

    qc.x(sol[1])
    qc.x(sol[2])
    qc.x(sol[3])

    qc.compose(MCMTGate(ZGate(), 3, 1), sol[:], inplace=True) 
    
    qc.x(sol[1])
    qc.x(sol[2])
    qc.x(sol[3])
    qc.barrier()

    # uncompute f
    qc.append(quad.inverse(), x[:] + sol[:])

    return qc

if __name__ == '__main__':
    nqb = 3
    nsol = 2
    ni = n_iters(nqb, nsol)
    print('niter: ', ni)
    shots = 1000
    sampler = Sampler()    
    
    # Grover's stuff
    oracle = get_oracle() 
    grover_op = GroverOperator(oracle, reflection_qubits = [0, 1, 2])
    oracle.draw(output="mpl", interactive=True)

    # circuit
    qc = QC(7, 3)
    
    qc.h([0, 1, 2])
    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()
    qc.measure([0, 1, 2], [0, 1, 2])

    qc.draw(output="mpl", interactive=True)
    
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = {k:v/shots for k, v in result[0].data['c'].get_counts().items()}
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()

