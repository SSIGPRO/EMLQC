from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import GroverOperator


def n_iters(nqb):
    np = 2**nqb
    ni = (pi/4)*sqrt(np)
    return floor(ni)

def get_oracle(nqb, target):
    qc = QC(nqb)

    # target è una stringa tipo "00", "01", "10", "11"

    if target == "00":
        qc.x(0)
        qc.x(1)
        qc.cz(0,1)
        qc.x(0)
        qc.x(1)

    elif target == "01":
        qc.x(0)
        qc.cz(0,1)
        qc.x(0)

    elif target == "10":
        qc.x(1)
        qc.cz(0,1)
        qc.x(1)

    elif target == "11":
        qc.cz(0,1)

    return qc


if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)

    for target in ["00", "01", "10", "11"]:

     print("\n======================")
     print("Target:", target)

    print('niter: ', ni)
    shots = 1000
    sampler = Sampler()    
    
    # Grover's stuff
    oracle = get_oracle(nqb,target) 
    grover_op = GroverOperator(oracle)
    
    # circuit
    qc = QC(nqb)
    
    #superposizione iniziale
    qc.h(np.arange(nqb))
    
    qc.barrier()

    
    # Grover iteratizione
    qc.compose(grover_op.power(ni), inplace=True)

    qc.barrier()
    
    qc.draw(output="mpl", interactive=True)

    qcm = qc.measure_all(inplace=False)
    job = sampler.run([qcm], shots=shots)
    result = job.result()

    
    counts = {k:v/shots for k, v in result[0].data['meas'].get_counts().items()}

    plot_histogram(counts)
    print(f" > Counts: {counts}")
    print(result)

    plt.show()