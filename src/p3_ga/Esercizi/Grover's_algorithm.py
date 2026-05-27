from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import GroverOperator

def n_iters(nqb):
    np = 2**nqb
    ni = (pi/4)*sqrt(np)
    return floor(ni)

def get_oracle_00(nqb):
   
    qc = QC(nqb)
    qc.x([0,1])
    qc.cz(0,1)
    qc.x([0,1])
    return qc

def get_oracle_01(nqb):
   
    qc = QC(nqb)
    qc.x(1)
    qc.cz(0,1)
    qc.x(1)
    return qc

def get_oracle_10(nqb):
   
    qc = QC(nqb)
    qc.x(0)
    qc.cz(0,1)
    qc.x(0)
    return qc

def get_oracle_11(nqb):
   
    qc = QC(nqb)
    qc.cz(0, 1) 
    return qc

if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    print('niter: ', ni)
    shots = 1000
    sampler = Sampler()    
    
    # Grover's stuff
    circuits = []
    for oracles in [get_oracle_00, get_oracle_01, get_oracle_10, get_oracle_11]:
        oracle = oracles(nqb) 
        grover_op = GroverOperator(oracle)
    
     # circuit
        qc = QC(nqb)
        qc.h(np.arange(nqb))

        qc.barrier()
        qc.compose(grover_op.power(ni), inplace=True)
        qc.barrier()

        qc.draw(output="mpl", interactive=True)
        qcm = qc.measure_all(inplace=False)
        circuits.append(qcm)

        job = sampler.run([qcm], shots=shots)
        result = job.result()
        counts = {k:v/shots for k, v in result[0].data['meas'].get_counts().items()}
        plot_histogram(counts)
        print(f" > Counts: {counts}")

    
    plt.show()

