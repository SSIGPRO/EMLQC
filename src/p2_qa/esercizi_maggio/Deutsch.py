from qiskit import QuantumCircuit as QC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

def f00(qc):
    return

def f01(qc):
    qc.cx(0, 1)
    return

def f10(qc):
    qc.x(0)
    qc.cx(0, 1)
    qc.x(0)
    return

def f11(qc):
    qc.x(1)
    return

if __name__ == '__main__':
    sampler = Sampler()

    qcs = []
    names = ["f00 ", "f01 ", "f10 ", "f11 "]

    for foo in [f00, f01, f10, f11]:
        qc = QC(2, 1)
        qc.initialize([1.0, 0.0], 0)
        qc.initialize([0.0, 1.0], 1)
        qc.h(0)
        qc.h(1)
        foo(qc)
        qc.h(0)
        qc.measure(qubit=0, cbit=0)
        qcs.append(qc)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Circuiti algoritmo di deutsch", fontsize=14)
    for ax, qc, name in zip(axes.flat, qcs, names):
        qc.draw(output="mpl", ax=ax)
        ax.set_title(name, fontsize=10)
    plt.tight_layout()

    job = sampler.run(qcs, shots=1000)
    results = job.result()

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 6))
    fig2.suptitle("Risultati algoritmo di deutsch ", fontsize=14)
    for ax, res, name in zip(axes2.flat, results, names):
        counts = res.data['c'].get_counts()
        result = max(counts, key=counts.get)
        kind = "CONSTANT" if result == "0" else "BALANCED"
        print(f"{name} -> {kind} (counts: {counts})")
        plot_histogram(counts, ax=ax, title=f"{name}\n→ {kind}")
    plt.tight_layout()

    plt.show()