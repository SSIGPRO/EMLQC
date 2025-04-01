# Preping up the environment

Clone the repo, given your current director to be `<cwd>`
```sh
git clone git@github.com:SSIGPRO/EMLQC.git
cd EMLQC
```

- Create and activate evironment
```sh
python -m venv emlqc
source emlqc/bin/activate
```

- In you are using `Miniconda` or `Conda`
```
conda create --name emlqc python
conda activate ./emlqc
```

- Install the dependencies
```sh
pip install torch gymnasium pygame
```

# Quantum Computing

- Install dependencies
```sh
pip install qiskit qiskit[visualization] PyQt6
```

