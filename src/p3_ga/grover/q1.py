from math import *
from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import GroverOperator
import matplotlib.pyplot as plt

def n_iters(nqb):
    return floor(pi / (4 * asin(sqrt(1 / 2**nqb))))

def oracle_for_10():
    
    # Crea un circuito con SOLO qubit (senza registri con nome)
    qc = QC(3, name="Oracle_10")  # 2 qubit dati + 1 ancilla
    
    # Marca lo stato |10⟩ (little-endian: q0=0, q1=1)
    qc.x(0)                 # Flip q0 (che è 0) → diventa 1
    qc.h(2)                 # H sull'ancilla
    qc.ccx(0, 1, 2)        # Toffoli (CCX) - più stabile di mcx
    qc.h(2)                 # H sull'ancilla
    qc.x(0)                 # Annulla flip
    
    return qc

if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    shots = 1024
    
    print(f"="*50)
    print(f"ALGORITMO DI GROVER")
    print(f"="*50)
    print(f"Problema: x + 1 = 3 → x = 2 → stato |10⟩")
    print(f"Qubit: {nqb} | Iterazioni: {ni} | Shots: {shots}")
    print(f"="*50)
    
    # Costruisci oracle e operatore di Grover
    oracle = oracle_for_10()
    grover_op = GroverOperator(oracle)
    
    # Circuito principale
    qc = QC(3, 2)  # 3 qubit totali, 2 bit classici
    qc.h([0, 1])   # Sovrapposizione sui 2 qubit di ricerca
    qc.barrier()
    qc.compose(grover_op, inplace=True)  # Applica Grover UNA volta
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    
    # Stampa circuiti in formato TESTO (per evitare errori matplotlib)
    print("\n CIRCUITO ORACLE:")
    print(oracle.draw(output='mpl'))
    
    print("\n CIRCUITO GROVER COMPLETO:")
    print(qc.draw(output='mpl'))
    
    # Esecuzione
    print("\n Esecuzione in corso...")
    sampler = Sampler()
    result = sampler.run([qc], shots=shots).result()
    counts = result[0].data['c'].get_counts()
    
    # Risultati
    print("\n📊 RISULTATI:")
    for state, count in sorted(counts.items(), key=lambda x: -x[1]):
        prob = count / shots

        marker = "✅ TARGET" if state == "10" else ""
        print(f"   |{state}⟩: {count:4d} / {shots} ({prob:.1%}) {marker}")
    
    # Istogramma
    print("\n GENERAZIONE ISTOGRAMMA...")
    plot_histogram(counts, title='Grover: ricerca x=2 (|10⟩)')
    plt.show()