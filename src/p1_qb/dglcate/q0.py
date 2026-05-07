from math import *

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':
    #commento l'inizializzazione di 1 qubit
    #qc = QC(1)
    #qc.initialize([1/sqrt(2),1/sqrt(2)], 0)
    #qc.draw(output="mpl", interactive=True)
    #plt.show()
    
    #inizializzo i due qubits insieme
    qc = QC(2)
    qc.initialize([sqrt(0.1), sqrt(0.2), sqrt(0.3), sqrt(0.4)], [0,1])
    qc.draw(output="mpl", interactive=True)
    plt.show()

    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data['meas'].get_counts()}")
    
