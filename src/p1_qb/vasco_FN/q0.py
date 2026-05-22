from math import sqrt, acos
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':
    # Creiamo un circuito con 2 qubit
    qc = QC(2)

    # 1. Impostiamo il primo qubit (q1)
    prob_q1_is_1 = 0.7
    theta1 = 2 * acos(sqrt(1 - prob_q1_is_1))
    qc.ry(theta1, 1)

    # 2. Impostiamo il secondo (q0) condizionato al primo
    theta_cond0 = 2 * acos(sqrt(1 - 2/3)) # Se q1=0
    theta_cond1 = 2 * acos(sqrt(1 - 4/7)) # Se q1=1

    qc.cry(theta_cond0, 1, 0, ctrl_state='0')
    qc.cry(theta_cond1, 1, 0, ctrl_state='1')

    # Visualizzazione
    print(qc.draw(output="text")) # Disegno veloce nel terminale
    
    # Esecuzione
    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    
    # Estrazione dei conteggi
    counts = result[0].data.meas.get_counts()
    print(f"\n > Counts finali (target 10, 20, 30, 40):")
    for stato in sorted(counts.keys()):
        percentuale = (counts[stato] / 1000) * 100
        print(f"Stato {stato}: {percentuale}%")
