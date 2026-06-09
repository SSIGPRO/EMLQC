from math import *

import torch
import numpy as np

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

# Define the functions f00, f01, f10, f11 as specified in the problem statement
def f00(qc):
    return

def f01(qc):
    qc.cx(0, 1)  # CNOT con controllo qubit 0, target qubit 1
    return

def f10(qc):
    qc.x(0)  # not su qubit 0
    qc.cx(0, 1)  # CNOT con controllo qubit 0, target qubit 1
    qc.x(0)  # not su qubit 0 per riportarlo allo stato iniziale
    return

def f11(qc):
    qc.x(1)  # not su qubit 1* (sempre)
    return
    
if __name__ == '__main__':
    sampler = Sampler()
qcs = []
for foo in [f00, f01, f10, f11]:
    
        qc = QC(2, 1)  # 2 qubit, 1 bit classico
        qc.initialize([1.0, 0.0], 0)  # Inizializza il primo qubit a |0>
        qc.initialize([0.0, 1.0], 1)  # Inizializza il secondo qubit a |1>

         # Applica porta Hadamard (H) a entrambi i qubit |0⟩ in (|0⟩+|1⟩)/√2 e |1⟩ in (|0⟩-|1⟩)/√2
        qc.h(0)  
        qc.h(1) 
        
        foo(qc)  
        qc.h(0) 

        
        # Misura il primo qubit 0 nel bit classico 0, lasciando il secondo qubit 1 non misurato
        qc.measure(qubit=0, cbit=0) 

        qc.draw(output="mpl", interactive=True)
        qcs.append(qc)
    
        job = sampler.run([qcs], shots=1000)
        result = job.result()
        
        for res in result:
         counts = res.data['c'].get_counts()
        plot_histogram(counts)
        
        print(f" > Counts: {counts}")
        plt.show()

