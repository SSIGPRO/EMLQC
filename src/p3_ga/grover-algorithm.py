from math import pi, sqrt, floor
import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit.circuit.library.grover_operator import grover_operator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

def n_iters(nqb, M=1):
    N = 2 ** nqb
    return floor((pi / 4) * sqrt(N / M))

def get_oracle(nqb: int, key: int):
    qc = QC(nqb)
    if not (key & 0b01):   
        qc.x(0)
    if not (key & 0b10):   
        qc.x(1)
    qc.cz(0, 1)
    if not (key & 0b01):
        qc.x(0)
    if not (key & 0b10):
        qc.x(1)
    return qc

def run_grover(nqb, key, shots=1000):
    ni = n_iters(nqb, M=1)
    oracle    = get_oracle(nqb, key)
    grover_op = grover_operator(oracle)

    qc = QC(nqb)
    qc.h(np.arange(nqb))
    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()

    fig_circ = qc.draw(output="mpl")
    fig_circ.suptitle(f"Circuito — target: |{key:02b}⟩", fontsize=11)

    qcm = qc.measure_all(inplace=False)
    sampler = Sampler()
    job    = sampler.run([qcm], shots=shots)
    result = job.result()

    all_states = [format(i, f'0{nqb}b') for i in range(2 ** nqb)]
    counts_raw = result[0].data["meas"].get_counts()
    counts     = {s: counts_raw.get(s, 0) / shots for s in all_states}
    return counts, ni, fig_circ

if __name__ == "__main__":
    nqb   = 2
    shots = 1000
    keys  = [0b00, 0b01, 0b10, 0b11]

    fig_hist, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig_hist.suptitle("Grover's Algorithm — ricerca delle 4 chiavi", fontsize=14)
    all_states = [format(i, f'0{nqb}b') for i in range(2 ** nqb)]

    for ax, key in zip(axes.flatten(), keys):
        counts, ni, fig_circ = run_grover(nqb, key, shots=shots)
        probs  = [counts[s] for s in all_states]
        colors = ['tomato' if s == format(key, f'0{nqb}b') else 'steelblue'
                  for s in all_states]
        ax.bar(all_states, probs, color=colors)
        ax.set_title(f"Target: |{key:02b}⟩  (iterazioni: {ni})")
        ax.set_xlabel("Stato")
        ax.set_ylabel("Probabilità")
        ax.set_ylim(0, 1.1)

    fig_hist.tight_layout()
    plt.show()