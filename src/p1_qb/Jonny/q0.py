from math import *

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':
    qc = QC(2)
    qc.initialize([1/sqrt(10), 1/sqrt(5), sqrt(3/10), sqrt(2/5)], [0, 1]) # this is the real initialization of both qubits together
    
    # Now, I'm summing the probabilities of the combinations where there is the value I'm looking for in order to get their distribution
    # (i.e., the "probability" that q0 is 0 is P(q0 = 0) = P(00)+P(10), etc.).
    # Doing so, the product of two probabilities won't be  equal to the exact probability of the combination of both (for example, 
    # P(q0 = 0) * P(q1 = 0) != P(00)); this is because the two qubits are not indipendent, they are entangled, so they cannot be separated. 
    # qc.initialize([sqrt(2/5),sqrt(3/5)], 0)
    # qc.initialize([sqrt(3/10),sqrt(7/10)], 1) 

    qc.draw(output="mpl", interactive=True)
    plt.show()

    qc_measured = qc.measure_all(inplace=False)
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=1000)
    result = job.result()
    print(f" > Counts: {result[0].data["meas"].get_counts()}")

