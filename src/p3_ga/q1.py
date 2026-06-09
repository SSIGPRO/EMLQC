from math import *
from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import GroverOperator

def n_iters(nqb):
    return floor(pi / (4 * asin(sqrt(1 / 2**nqb))))

def oracle_for_10():
    """Oracle corretto che marca lo stato |10⟩ (x=2)"""
    qc = QC(3)  # 2 qubit dati + 1 ancilla
    
    # Step 1: Flip dove il bit target è 0
    # Per |10⟩ con little-endian: q0=0 (LSB), q1=1 (MSB)
    qc.x(0)     # q0 è 0 → lo flip a 1
    
    # Step 2: Multi-controlled Z sull'ancilla
    qc.h(2)                     # Prepara ancilla
    qc.mcx([0, 1], 2)           # Toffoli: flip ancilla se entrambi sono 1
    qc.h(2)                     # Trasforma in Z controllata
    
    # Step 3: Annulla i flip
    qc.x(0)
    
    return qc

if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    
    oracle = oracle_for_10()
    grover_op = GroverOperator(oracle)
    
    qc = QC(3, 2)
    qc.h([0, 1])
    qc.compose(grover_op.power(ni), inplace=True)
    qc.measure([0, 1], [0, 1])
    
    # Esegui
    sampler = Sampler()
    result = sampler.run([qc], shots=1024).result()
    counts = result[0].data['c'].get_counts()
    
    print(f"Risultati: {counts}")
    print("Mi aspetto che '10' sia lo stato più probabile")
    plot_histogram(counts)