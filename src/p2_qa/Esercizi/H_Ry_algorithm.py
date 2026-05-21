from math import *

from matplotlib import pyplot as plt

from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

def f0(qc):
    qc.h(0)
    return

def f1(qc):
    qc.ry(pi/2,0)
    return

if __name__ == '__main__':
    sampler = Sampler()
    shots = 1   
    
    qcs = []
    for foo in [f0, f1]:
        qc = QC(1, 1)
        qc.initialize([1.0, 0.0], 0)
        qc.h(0)
        foo(qc)
        qc.ry(pi/2,0)
        qc.h(0)
        qc.measure(qubit=0,cbit=0)
        qc.draw(output="mpl", interactive=True)
        qcs.append(qc)

    job = sampler.run(qcs, shots=1000)
    results = job.result()
    for res in results:
        counts = res.data['c'].get_counts()
        plot_histogram(counts)
        print(f" > Counts: {counts}")
    plt.show()

