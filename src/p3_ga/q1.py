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

def n_iters(nqb):
    ni = pi / (4 * asin(sqrt(1 / 2**nqb)))
    return floor(ni)

def get_oracle():
    a = QuantumRegister(2, 'a')
    b = QuantumRegister(2, 'b')
    c = QuantumRegister(2, 'c')

    qc = QC(*[a, b, c])

    qc.x(a[1])

    qc.h(b)

    qc.x(c)
    qc.barrier()

    adder = Add(2, kind='fixed', name='Add')

    qc.append(adder, a[:]+b[:])
    qc.barrier()

    qc.compose(MCMTGate(ZGate(), 1, 1), b[:], inplace=True)
    qc.barrier()

    qc.append(adder.inverse(), a[:]+b[:])

    qc.h(b)
    qc.barrier()

    return qc

if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    print('niter: ', ni)
    shots = 1000
    sampler = Sampler()

    oracle = get_oracle()
    grover_op = GroverOperator(oracle, reflection_qubits=[2, 3])
    oracle.draw(output="mpl", interactive=True)

    qc = QC(6, 2)

    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()
    qc.measure([2, 3], [0, 1])

    qc.draw(output="mpl", interactive=True)

    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = {k: v/shots for k, v in result[0].data['c'].get_counts().items()}
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()