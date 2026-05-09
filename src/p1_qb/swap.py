from math import *
import torch
import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit import transpile
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    sampler = Sampler()    
    num_samples = 10000
    
    c0 = np.sqrt(np.array([0.8, 0.2]))
    c1 = np.sqrt(np.array([0.3, 0.7]))

    qc = QC(2, 2)
    qc.initialize(c0, 0)
    qc.initialize(c1, 1)
    
    qc_measured = qc.measure_all(inplace=False, add_bits=False)
    
    qc.swap(0, 1)
    
    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=0, cbit=0)
    
    qc.draw(output="mpl", interactive=True)
    
    job_before = sampler.run([qc_measured], shots=num_samples)
    
    result_before = job_before.result()
    counts_before = result_before[0].data["c"].get_counts()
    
    probs_before = {
        stato: conteggio / num_samples  
        for stato, conteggio in counts_before.items()
    }
    plot_histogram(probs_before, title="circuito prima dello swap")
    
    job = sampler.run([qc], shots=num_samples)
    result = job.result()
    print(result)
    print(result[0].data)
    counts_after = result[0].data["c"].get_counts()
    
    probs_after = {
        stato: conteggio / num_samples  
        for stato, conteggio in counts_after.items()
    }
    plot_histogram(probs_after, title="circuito dopo lo swap")
    
    print(f" > Counts: {counts_after}")
    plt.show()
