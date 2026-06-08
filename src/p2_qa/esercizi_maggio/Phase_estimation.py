import torch
import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT

def apply_phase_gate(circuit, phi, ctrl, targ):
    circuit.crz(2 * np.pi * phi, ctrl, targ)
    return circuit

def build_circuit(phi):
    system = QuantumCircuit(3, 2)

    system.h(0)
    system.h(1)
    system.x(2) 
    system.barrier()

    apply_phase_gate(system, phi, 0, 2)
    apply_phase_gate(system, phi, 1, 2)
    apply_phase_gate(system, phi, 1, 2)
    system.barrier()
    
    system.compose(
        QFT(2, inverse=True).decompose(),
        inplace=True
    )
    return system



if __name__ == '__main__':
    backend_sampler = StatevectorSampler()    
    total_shots = 1000
    phi = 1/4
   
    sim_circuit = build_circuit(phi)

    meas_circuit = build_circuit(phi)
    meas_circuit.measure(0, 0)
    meas_circuit.measure(1, 1)

    meas_circuit.draw(output="mpl", interactive=True)

    job_execution = backend_sampler.run([meas_circuit], shots=total_shots)
    raw_results = job_execution.result()
    output_counts = raw_results[0].data['c'].get_counts()
    
    plot_histogram({bit: count / total_shots for bit, count in output_counts.items()})

    plot_bloch_multivector(Statevector(sim_circuit))
    plot_state_qsphere(sim_circuit)
    plt.show()