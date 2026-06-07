from math import floor, pi, sqrt

from matplotlib import pyplot as plt
from numpy.lib.scimath import arcsin
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit as QC
from qiskit.circuit.library.arithmetic.quadratic_form import (
    QuadraticFormGate,
)
from qiskit.circuit.library.grover_operator import grover_operator
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

n_input = 3
n_output = 4

quadratic = [[1, 2, 4], [2, 4, 8], [4, 8, 16]]
linear = [-2, -4, -8]
offset = 1


def n_iters(n_qubit, n_solutions=1):
    N = 2**n_qubit
    theta = arcsin(sqrt(n_solutions / N))
    ni = pi / (4 * theta)
    return floor(ni)


def get_oracle():
    input_reg = QuantumRegister(n_input, "input")
    output_reg = QuantumRegister(n_output, "output")
    qc = QC(*[input_reg, output_reg])
    gate = QuadraticFormGate(n_output, quadratic, linear, offset)  # ty:ignore[invalid-argument-type]
    qc.append(gate, input_reg[:] + output_reg[:])
    qc.x(output_reg[3])
    qc.cz(output_reg[0], output_reg[3])
    qc.x(output_reg[3])
    qc.append(gate.inverse(), input_reg[:] + output_reg[:])
    return qc


if __name__ == "__main__":
    n_iterations = n_iters(n_input, 2)
    print("Number of iterations: ", n_iterations)
    shots = 1000
    sampler = Sampler()
    input_reg = QuantumRegister(n_input, "input")
    output_reg = QuantumRegister(n_output, "output")
    c = ClassicalRegister(n_input, "c")
    oracle = get_oracle()
    oracle.draw(output="mpl")
    grover_op = grover_operator(oracle, reflection_qubits=input_reg)
    qc = QuantumCircuit(*[input_reg, output_reg, c])
    qc.h(input_reg)
    qc.barrier()
    qc.compose(grover_op.power(n_iterations), inplace=True)
    qc.barrier()
    qc.draw(output="mpl", interactive=True)
    qc.measure(input_reg, c)
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = {k: v / shots for k, v in result[0].data["c"].get_counts().items()}
    plot_histogram(counts)
    print(f" > Counts: {counts}")
    plt.show()
