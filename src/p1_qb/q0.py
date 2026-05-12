from math import *

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':
    qc = QC(2)
    #qc.initialize([sqrt(0.1),sqrt(0.2),sqrt(0.3),sqrt(0.4)], [0,1]) Qubit inizializzati insieme
    qc.initialize([sqrt(0.29),sqrt(0.71)], 0) 
    qc.initialize([sqrt(0.43),sqrt(0.57)], 1) #qubit inizializzati separatamente
    qc.draw(output="mpl", interactive=True)
    plt.show()

    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data['meas'].get_counts()}")

