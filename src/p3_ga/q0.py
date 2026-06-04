from math import floor, pi, sqrt

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit.circuit.library.grover_operator import grover_operator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram


def n_iters(nqb):
    np = 2**nqb
    ni = (pi / 4) * sqrt(np)
    return floor(ni)


def get_oracle(nqb: int, key: int):
    qc = QC(nqb)
    if key == 0b00 or key == 0b01:
        qc.x(0)
    if key == 0b00 or key == 0b10:
        qc.x(1)
    qc.cz(0, 1)
    if key == 0b00 or key == 0b01:
        qc.x(0)
    if key == 0b00 or key == 0b10:
        qc.x(1)
    return qc


if __name__ == "__main__":
    nqb = 2
    key = 0b11
    ni = n_iters(nqb)
    print("niter: ", ni)
    shots = 1000
    sampler = Sampler()

    # Grover's stuff
    oracle = get_oracle(nqb, key)
    grover_op = grover_operator(oracle)

    # circuit
    qc = QC(nqb)
    qc.h(np.arange(nqb))

    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()

    qc.draw(output="mpl", interactive=True)

    qcm = qc.measure_all(inplace=False)
    job = sampler.run([qcm], shots=shots)  # ty:ignore[invalid-argument-type]
    result = job.result()
    counts = {k: v / shots for k, v in result[0].data["meas"].get_counts().items()}
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()
