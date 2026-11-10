# qfim-operator-selection-cs-vqe
Our submission for Qiskit Fall Fest 2026! There are a lot of files in this repository, so here's how to navigate and run the project:

## How to Run

### 1. Setting up environment

The project uses Python, Qiskit, NumPy, SciPy, pandas, Matplotlib, Symmer, and PySCF. Hardware experiments additionally require Qiskit IBM Runtime and access to IBM Quantum.

### 2. Extracting input data

Molecular input files are provided in `data.zip` due to GitHub's browser upload limit. Please download and extract `data.zip` into the repository's root directory. The extracted folder should be named `data/`.

### 3. Reproducing simulation benchmarks

Run the main simulation script from the repository's root directory:

```bash
python "[qfim]run_vqe_cs_sqd.py"
```

The saved simulation results are also included in `benchmark_results.csv`, so rerunning the VQE optimizations is not necessary to view the reported results.

### 4. Viewing results

Open `Project_Notebook.ipynb` in JupyterLab or any Jupyter-compatible editor.

The notebook uses the saved benchmark results to display the VQE accuracy comparison, circuit resources, and preliminary hardware results.

### 5. Hardware comparison

The repository includes two hardware scripts:

- `[qfim]run_vqe_cs_sqd_hardware.py`: QFIM-selected VQE hardware experiment.
- `[qfim]compare_fixed_hardware.py`: Fixed-parameter comparison between QFIM- and gradient-selected circuits.

These scripts require configured IBM Quantum credentials and submit jobs to real quantum hardware, which may consume runtime credits. BUT! The results of the fixed-parameter comparison are already saved in `fixed_hardware_results/`, so hardware access is not needed to review the reported results.
