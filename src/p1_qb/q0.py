from math import *

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler

if __name__ == "__main__":
    qc1 = QC(2)
    qc1.initialize([sqrt(0.1), sqrt(0.2), sqrt(0.3), sqrt(0.4)], [0, 1])
    qc1.draw(output="mpl", interactive=True)
    plt.show()
    qc1_measured = qc1.measure_all(inplace=False)
    sampler1 = StatevectorSampler()
    job1 = sampler1.run([qc1_measured], shots=1000)
    result1 = job1.result()
    print(f" > Counts: {result1[0].data['meas'].get_counts()}")
    qc2 = QC(2)
    # To initialize the two qubits separately we need to change the probability.
    # I choose alpha_00 = sqrt(1/12) alpha_01 = sqrt(3/12) alpha_10 = sqrt(2/12) alpha_11 = sqrt(6/12)
    # This lead to P(|00>) = 8.33% P(|01>) = 25.00% P(|10>) = 16.67% P(|11>) = 50.00%
    # Known that the event are independent we can write P(|00>)=P(Q_1 = |0>)*P(Q_2 = |0>) and so on
    # Leading to beta_0*gamma_0=sqrt(1/12); beta_0*gamma_1=sqrt(3/12); beta_1*gamma_0=sqrt(2/12); beta_1*gamma_1=sqrt(6/12);
    # Adding beta_0^2+beta_1^2=1; gamma_0^2+gamma_1^2=1 to the system and solving it leads to the following coefficients
    #
    # NOTE: To see the second diagram close the first one
    qc2.initialize([1 / sqrt(3), sqrt(2 / 3)], 0)
    qc2.initialize([1 / 2, sqrt(3) / 2], 1)
    qc2.draw(output="mpl", interactive=True)
    plt.show()
    qc2_measured = qc2.measure_all(inplace=False)
    sampler2 = StatevectorSampler()
    job2 = sampler2.run([qc2_measured], shots=1000)
    result2 = job2.result()
    print(f" > Counts: {result2[0].data['meas'].get_counts()}")
