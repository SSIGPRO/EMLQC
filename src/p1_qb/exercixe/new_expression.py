from math import *
from qiskit import QuantumCircuit as QC, QuantumRegister
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import GroverOperator, MCMTGate, ZGate
from qiskit.circuit.library import QuadraticFormGate as QuadraticForm 

def n_iters(nqb):
    ni = pi / (4 * asin(sqrt(1 / 2**nqb)))
    #ni = (pi/4)*sqrt(2**nqb)
    return floor(ni)

def get_oracle():
    reg_a = QuantumRegister(1, 'reg_a')
    reg_b = QuantumRegister(2, 'reg_b')
    qc = QC(*[reg_a, reg_b])

    qc.h(reg_b)
    qc.barrier() 

    A = [[1]]
    linear = [-2]
    q_form = QuadraticForm(2, A, linear)                            
    
    qc.append(q_form, reg_a[:] + reg_b[:])
    qc.barrier()

    qc.h(reg_b[1])
    qc.compose(MCMTGate(ZGate(), 1, 1), reg_b[:], inplace=True)
    qc.x(reg_b)
    qc.barrier()

    qc.append(q_form.inverse(), reg_a[:] + reg_b[:])
    
    qc.h(reg_b)
    qc.barrier()
    
    return qc

if __name__ == '__main__':
    nqb = 2
    ni = n_iters(nqb)
    print('niter: ', ni)
    shots = 10000
    sampler = Sampler()    
    
    # Grover's stuff
    oracle = get_oracle() 
    grover_op = GroverOperator(oracle, reflection_qubits = [1, 2])
    oracle.draw(output="mpl", interactive=True)

    qc = QC(3, 2)
    
    qc.barrier()
    qc.compose(grover_op.power(ni), inplace=True)
    qc.barrier()
    qc.measure([1, 2], [0, 1])

    qc.draw(output="mpl", interactive=True)
    
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = {k:v/shots for k, v in result[0].data['c'].get_counts().items()}
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()

