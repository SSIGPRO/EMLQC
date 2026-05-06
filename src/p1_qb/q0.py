from math import sqrt
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler

if __name__ == '__main__':
    qc = QC(2)
    
    num_samples = 5000 #numero di misurazioni da effettuare
    
    state = [sqrt(0.10), sqrt(0.20), sqrt(0.30), sqrt(0.40)]
    
    qc.initialize(state, [0, 1])
    
    qc.draw(output="mpl", interactive=True)
    plt.show()

    qc_measured = qc.measure_all(inplace=False)
    
    sampler = StatevectorSampler()    
    job = sampler.run([qc_measured], shots=num_samples)
    result = job.result()
    
    counts = result[0].data['meas'].get_counts()

    print("Valori conteggi:")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")

    print("Valori percentuali:")
    for key in sorted(counts):
        print(f"{key}: {counts[key]/(num_samples/100):.2f}%")
