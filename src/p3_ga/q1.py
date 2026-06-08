from math import pi, floor, asin, sqrt
import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import GroverOperator, QuadraticForm

def n_iters(nqb, n_sol=1):
    return floor(pi / (4 * asin(sqrt(n_sol / 2**nqb))))

def get_oracle(n_x: int, n_res: int):
    x   = QuantumRegister(n_x,  'x')
    res = QuantumRegister(n_res, 'res')
    qc  = QC(x, res)

    A = [[0, 1],
         [0, 1]]
    b = [-1, -2]
    c = 0

    qf = QuadraticForm(
        num_result_qubits = n_res,
        quadratic = A,
        linear    = b,
        offset    = c,
        little_endian = True
    )
    
    qc.append(qf, x[:] + res[:])
    qc.barrier()

    qc.x(res)
    qc.h(res[-1])
    qc.mcx(res[:-1], res[-1])   
    qc.h(res[-1])
    qc.x(res)
    qc.barrier()

    qc.append(qf.inverse(), x[:] + res[:])

    return qc

if __name__ == '__main__':
    n_x   = 2
    n_res = 3
    n_sol = 2

    ni    = n_iters(n_x, n_sol)
    shots = 1000
    
    print(f"(x-1)^2 = 1  →  soluzioni: x=0, x=2")
    print(f"n_iter: {ni}")

    sampler = Sampler()

    oracle     = get_oracle(n_x, n_res)
    grover_op  = GroverOperator(oracle, reflection_qubits=list(range(n_x)))

    total_qb = n_x + n_res
    
    c_reg = ClassicalRegister(n_x, 'c')
    qc = QC(QuantumRegister(total_qb, 'q'), c_reg)
    
    qc.h(range(n_x))
    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()
    qc.measure(range(n_x), range(n_x))

    job    = sampler.run([qc], shots=shots)
    result = job.result()
    
    counts = {k: v/shots for k, v in result[0].data.c.get_counts().items()}

    decimal_counts = {f"|{b}⟩ x={int(b,2)}": p for b, p in counts.items()}
    print(f"Counts: {decimal_counts}")

    plot_histogram(decimal_counts, title="(x-1)²=1 — Grover + QuadraticForm")
    plt.show()