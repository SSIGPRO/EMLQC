from math import *

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':

    # METODO STANDARD DOVE DICHIARO LA PROBABILITA' DELLE COMBINAZIONE
    qc = QC(2)
    qc.initialize([sqrt(1/10), sqrt(2/10), sqrt(3/10), sqrt(4/10)], [0 , 1])
    
    # METODO CHE DICHIARA LA PROBABILITA' DEI SINGOLI QUBIT DI AVERE 1 o 0
    # Il metodo non funziona completamente, si avvicina alle probabilità ma non le aggiunge completamente.
    # Inizializzando i qubit così non rispettiamo la proprietà di entanglment dei due qubit, dato che le probabilità date fanno riferimento alla coppia
    # Probabilità: 00 -> 12%  01 -> 18%  10 -> 28%  11 -> 42%
    # Le probabilità di essere 0 o 1 dei singoli qubit le ho ricavate calcolando la probabilità marginale

    # qc = QC(2)
    # qc.initialize([sqrt(3/10), sqrt(7/10)], 0)
    # qc.initialize([sqrt(4/10), sqrt(6/10)], 1)

    qc.draw(output="mpl", interactive=True)
    plt.show()

    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data['meas'].get_counts()}")

