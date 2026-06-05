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

def get_oracle(nqb, key):

    if key == 1:          #key = 00
        qc = QC(nqb)
        qc.x(0)
        qc.x(1)
        qc.cz(1,0)
        qc.x(0)
        qc.x(1)        
        return qc
    
    if key == 2:          #key = 01
        qc = QC(nqb)
        qc.x(0)
        qc.cz(0,1)
        qc.x(0)
        return qc
    
    if key == 3: 
        qc = QC(nqb)
        qc.x(0)
        qc.cz(0,1)         #key = 10
        return qc
    
    if key == 4:          #key = 11
        qc = QC(nqb)
        qc.x(0)
        qc.cz(0,1)
        qc.x(1)
        return qc
    

if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    print('niter: ', ni)
    shots = 1000
    sampler = Sampler()    
    
    for key in range(1,5):
        # Grover's stuff
        oracle = get_oracle(nqb,key) 
        grover_op = GroverOperator(oracle)
        
        # circuit
        qc = QC(nqb)
        qc.h(np.arange(nqb))

        qc.barrier()
        qc.compose(grover_op.power(ni), inplace=True)
        qc.barrier()

        qc.draw(output="mpl", interactive=True)
        
        qcm = qc.measure_all(inplace=False)
        job = sampler.run([qcm], shots=shots)
        result = job.result()
        counts = {k:v/shots for k, v in result[0].data['meas'].get_counts().items()}
        plot_histogram(counts)
        print(f" > Counts: {counts}")
    plt.show()

