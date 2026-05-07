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
    #ora commentato, per l'inizializzazione dei due qubit separatamente
    #qc = QC(2)
    #qc.initialize([sqrt(0.1), sqrt(0.2), sqrt(0.3), sqrt(0.4)], [0,1])
    #qc.draw(output="mpl", interactive=True)
    #plt.show()

    #inizializzo i due qubit separatamente
    qc = QC(2)
    qc.initialize([sqrt(0.1+0.2), sqrt(0.3+0.4)], 0)
    qc.initialize([sqrt(0.1+0.3), sqrt(0.2+0.4)], 1)
    qc.draw(output="mpl", interactive=True)
    plt.show()
    #commenti sul metodo:
    #non si riesce a iniziallizare i due qubit separatamente con un metodo
    #che sia esattamente equivalente all'inizializzazione contemporanea
    #Però questo metodo vi si avvicina:
    #sono partita dal calcolare, per ogni qubit, la probabilità che esso sia o 0 o 1:
    #trovando che P(|q_0>=0)=0.1+0.2=0.3, P(|q_0>=1)=0.3+0.4=0.7, P(|q_1>=0)=0.1+0.3=0.4, P(|q_1>=1)=0.2+0.4=0.6
    #Poi ho calcolato la probabilità del sistema formato dai due qubit combinati, calcolando la probabilità di essere in un determinato stato:
    #P(|00>)=0.3*0.4=0.12, P(|01>)=0.3*0.6=0.18, P(|10>)=0.7*0.4=0.28, P(|11>)=0.7*0.6=0.42
    #Le probabilità ottetnute (12%, 18%, 28%, 42%) differiscono dalle probabilità target (10%, 20%, 30%, 40%), ma vi si avvicinano
    #Inoltre, mettendo a sistema 4 equazioni in 4 incognite (una per ogni probabilità di ogni qubit di essere o 0 o 1), cioè i candidati a essere i coefficienti alfa^2,
    #ed eguagliando ogni equazione alla probabilità target, il sistema risulta impossibile

    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data['meas'].get_counts()}")
    
