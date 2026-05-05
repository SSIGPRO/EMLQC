from math import *

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':
    qc = QC(2)
    
    # inizializzazione 2 qubit insieme
    qc.initialize([sqrt(1/10),sqrt(2/10),sqrt(3/10),sqrt(4/10)], [0,1])

    # inizializzazione 2 qubit separati
    #qc.initialize([sqrt(6/10),sqrt(4/10)], 0)
    #qc.initialize([sqrt(2/10),sqrt(8/10)], 1)

    qc.draw(output="mpl", interactive=True)
    plt.show()

    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data["meas"].get_counts()}")

