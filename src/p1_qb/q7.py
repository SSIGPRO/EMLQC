from math import pi
import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.visualization import (
    plot_histogram,
    plot_bloch_multivector,
    plot_state_qsphere,
)
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT


#  PREPARAZIONE AUTOSTATO DI RY

def prepara_autostato_ry(circuito, qubit_target, autovalore="+"):
    circuito.ry(pi / 2, qubit_target)   # |0> → |+y>
    if autovalore == "-":
        circuito.z(qubit_target)        # |+y> → |-y>



#  QPE PER RY(θ)

def qpe_ry(frazione_theta, n_contatori=3, autovalore="+"):
    circuito = QuantumCircuit(n_contatori + 1, n_contatori)

    # prepara autostato
    prepara_autostato_ry(circuito, n_contatori, autovalore)
    circuito.barrier()

    # Hadamard sui qubit di conteggio
    for q in range(n_contatori):
        circuito.h(q)
    circuito.barrier()

    # applicazioni controllate di RY(θ)
    ripetizioni = 1
    for i in range(n_contatori):
        for _ in range(ripetizioni):
            angolo = 2 * pi * frazione_theta
            circuito.cry(angolo, i, n_contatori)
        ripetizioni *= 2
    circuito.barrier()

    # QFT inversa
    circuito.compose(
        QFT(n_contatori, inverse=True).decompose(),
        inplace=True,
        qubits=range(n_contatori)
    )

    return circuito


#  QPE PER HADAMARD

def qpe_hadamard(n_contatori=3):
    circuito = QuantumCircuit(n_contatori + 1, n_contatori)

    # autostato di H: |+>
    circuito.h(n_contatori)
    circuito.barrier()

    # Hadamard sui contatori
    for q in range(n_contatori):
        circuito.h(q)
    circuito.barrier()

    # applicazioni controllate di H
    ripetizioni = 1
    for i in range(n_contatori):
        for _ in range(ripetizioni):
            circuito.ch(i, n_contatori)
        ripetizioni *= 2
    circuito.barrier()

    # QFT inversa
    circuito.compose(
        QFT(n_contatori, inverse=True).decompose(),
        inplace=True,
        qubits=range(n_contatori)
    )

    return circuito


#  INTERPRETAZIONE RISULTATI QPE

def interpreta_risultato(counts, n_contatori):
    bit_piu_probabile = max(counts, key=counts.get)
    fase_decimale = int(bit_piu_probabile, 2) / (2 ** n_contatori)
    fase_radianti = fase_decimale * 2 * pi
    return fase_radianti, bit_piu_probabile

#  PROGRAMMA PRINCIPALE
if __name__ == "__main__":
    campionatore = Sampler()
    colpi = 4096
    n_contatori = 4

    print("QPE PER RY(θ) e HADAMARD")
    print(f"Precisione angolare: {360 / (2 ** n_contatori):.1f}°")

    # test per RY
    lista_angoli = [
        (0, "+"),
        (1/8, "+"),
        (1/4, "+"),
        (1/2, "+"),
        (1/4, "-"),
    ]

    for frazione_theta, autovalore in lista_angoli:
        print(f"\n--- RY(θ={frazione_theta:.3f}×2π, autovalore {autovalore}) ---")

        circuito = qpe_ry(frazione_theta, n_contatori, autovalore)

        # misura i qubit di conteggio
        for q in range(n_contatori):
            circuito.measure(q, q)

        # esecuzione
        risultato = campionatore.run([circuito], shots=colpi).result()
        quasi = risultato.quasi_dists[0]
        counts = quasi.binary_probabilities()

        fase, bit = interpreta_risultato(counts, n_contatori)

        print(f"Bit più probabile: {bit} → fase {fase:.3f} rad ({fase * 180 / pi:.1f}°)")

        # mostra i 3 risultati più probabili
        top3 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for esito, prob in top3:
            print(f"  {esito}: {prob:.1%}")

        plot_histogram(counts, title=f"RY({frazione_theta:.2f}×2π)")
        plt.show()

    # test per Hadamard
    print("\n--- HADAMARD (H) ---")

    circuito_h = qpe_hadamard(n_contatori)
    for q in range(n_contatori):
        circuito_h.measure(q, q)

    risultato_h = campionatore.run([circuito_h], shots=colpi).result()
    quasi_h = risultato_h.quasi_dists[0]
    counts_h = quasi_h.binary_probabilities()

    top4 = sorted(counts_h.items(), key=lambda x: x[1], reverse=True)[:4]
    for esito, prob in top4:
        fase = int(esito, 2) / (2 ** n_contatori) * 2 * pi
        print(f"{esito} (fase {fase:.2f} rad): {prob:.1%}")

    plot_histogram(counts_h, title="Hadamard (H)")
    plt.show()

    # visualizzazione stato
    circuito_vis = qpe_ry(1/4, n_contatori, "+")
    stato = Statevector.from_instruction(circuito_vis)

    plot_bloch_multivector(stato)
    plot_state_qsphere(stato)
    plt.show()
