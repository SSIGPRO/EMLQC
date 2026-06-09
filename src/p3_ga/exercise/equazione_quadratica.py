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
from qiskit.circuit.library import QuadraticFormGate as QFG

def n_iters(nqb):
    ni = pi / (4 * asin(sqrt(1 / 2**nqb)))
    #ni = (pi/4)*sqrt(2**nqb)
    return floor(ni)

def get_oracle():
    a = QuantumRegister(1, 'a')
    b = QuantumRegister(2, 'b')

    qc = QC(*[a, b])

    # First H from marking 
    qc.h(b)

    qc.barrier() 

    #Initialize the quadratic function
    A= [[1]]
    linear= [-2]
    quadratic = QFG(2, A, linear)                            #Quadratic Form Gate

    qc.append(quadratic, a[:]+b[:])
    qc.barrier() 

    # if (x-1)^2 == 1, the output (on qubits b) will be 00
    # Split equally b[1] and invert phase on 00 in order to have 50% 0 and 50% 2 
    qc.h(b[1])
    qc.compose(MCMTGate(ZGate(), 1, 1), b[:], inplace=True) 
    qc.x(b)
    qc.barrier()

    # uncompute f
    qc.append(quadratic.inverse(), a[:]+b[:])
    # Last H from the marking
    qc.h(b)
    qc.barrier()

    return qc

if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    print('niter: ', ni)
    shots = 10000
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
