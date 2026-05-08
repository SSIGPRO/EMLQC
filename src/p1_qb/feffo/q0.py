#fare esercizio precedente con 10/20/30/40
#se facciamo divisi i due qubits un punto in più 
# 0 0 1 1 
# 0 1 0 1 


from math import *

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':
    
     #inizializzazione singola
    p_00 = 0.1
    p_01 = 0.2
    p_10 = 0.3 
    p_11 = 0.4 

    qc_singola = QC(2)
    qc_singola.initialize([sqrt(p_00),sqrt(p_01),sqrt(p_10),sqrt(p_11)], [0,1])
    qc_singola.draw(output="mpl", interactive=True)
    plt.show()

    #inizializazione separata
        #calcolo algebrico delle probabilità marginali 
            #qubit 0 
    pm_00 = p_00 + p_10
    pm_01 = p_01 + p_11 

            #qubit 1
    pm_10 = p_00 + p_01
    pm_11 = p_10 + p_11

    qc_separata = QC(2)
    qc_separata.initialize([sqrt(pm_00),sqrt(pm_01)], 0 )
    qc_separata.initialize([sqrt(pm_10),sqrt(pm_11)], 1 )
    qc_separata.draw(output="mpl", interactive=True)
    plt.show()




    qc_singola_measured = qc_singola.measure_all(inplace=False)
    qc_separata_measured = qc_separata.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_singola_measured,qc_separata_measured], shots=1000)
    result = job.result()
    print(f" > Counts (singola): {result[0].data['meas'].get_counts()}")
    print(f" > Counts (separata): {result[1].data['meas'].get_counts()}")

