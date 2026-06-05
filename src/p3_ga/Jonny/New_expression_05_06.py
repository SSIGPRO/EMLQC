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
from qiskit.circuit.library import UGate

def n_iters(nqb):
    ni = pi / (4 * asin(sqrt(1 / 2**nqb)))
    #ni = (pi/4)*sqrt(2**nqb)
    return floor(ni)

def get_oracle():
    a = QuantumRegister(2, 'a')
    b = QuantumRegister(2, 'b')
    c = QuantumRegister(2, 'c')

    qc = QC(*[a, b, c])

    # make a = b'10 = d'2
    qc.x(a[1])

    # First H from marking 
    qc.h(b)

    # make c = b'11b, d'3
    qc.x(c[1])
    qc.barrier() 
    
    #adder = Add(2, kind='fixed', name='Add')
    suber0 = Add(2, kind='fixed', name='Sub').inverse()
    u_gate = UGate(0, 0, 0).power(0.5, annotated= False)
    suber1 = Add(2, kind='fixed', name='Sub').inverse()

    qc.append(suber0, a[:]+b[:])
    qc.append(u_gate, b[:])
    qc.append(suber1, c[:]+b[:])
    qc.barrier() 

    # if the a+b == c, the output (on qubits b will be 00)
    # Invert phase on 00 
    qc.x(b)
    qc.compose(MCMTGate(ZGate(), 1, 1), b[:], inplace=True) 
    qc.x(b)
    qc.barrier()

    # uncompute f
    qc.append(suber1.inverse(), c[:]+b[:])
    qc.append(suber0.inverse(), a[:]+b[:])
    # Last H from the marking
    qc.h(b)
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
    grover_op = GroverOperator(oracle, reflection_qubits = [2, 3])
    oracle.draw(output="mpl", interactive=True)

    # circuit
    qc = QC(6, 2)
    
    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()
    qc.measure([2, 3], [0, 1])

    qc.draw(output="mpl", interactive=True)
    
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = {k:v/shots for k, v in result[0].data['c'].get_counts().items()}
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()

