from qiskit import QuantumCircuit as QC
import numpy as np
from qiskit.circuit.library import GroverOperator
from qiskit.primitives 
from math import floor, pi, sqrt


def n_iters(nqb):
    N = 2 ** nqb
    M = 4
    return floor((pi / 4) * sqrt(N / M))


def get_oracle(nqb):
    qc = QC(nqb)

    qc.cz(0, 1)

    qc.x(1)
    qc.cz(0, 1)
    qc.x(1)

    qc.x(0)
    qc.cz(0, 1)
    qc.x(0)

    qc.x(0)
    qc.x(1)
    qc.cz(0, 1)
    qc.x(0)
    qc.x(1)

    return qc


if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    print('niter: ', ni)
    shots = 1000
    sampler = Sampler()

    oracle = get_oracle(nqb)
    grover_op = GroverOperator(oracle)

    qc = QC(nqb)
    qc.h(np.arange(nqb))

    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()

    qc.measure_all()

    job = sampler.run(qc, shots=shots)
    result = job.result()
    counts = result.quasi_dists[0]

    print('Risultati (stato: probabilità):')
    for state, prob in sorted(counts.items()):
        label = format(state, f'0{nqb}b')
        print(f'  |{label}⟩ : {prob:.4f}')
