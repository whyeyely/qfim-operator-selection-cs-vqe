import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_original_code():
    path = ROOT / '[qfim]run_vqe_cs_sqd_hardware.py'
    if not path.is_file():
        raise FileNotFoundError(f'Place this script beside {path.name}')
    spec = importlib.util.spec_from_file_location('qfim_hardware_code', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_saved_result(data_dir, method):
    folder = Path(data_dir) / 'cs_vqe_sqd' / f'sv-{method}-0.25-sqd-cs'
    path = folder / 'HCl_STO-3G_SINGLET_JW.json'
    if not path.is_file():
        raise FileNotFoundError(
            f'Cannot find {method} statevector result at:\n{path}\n'
            'Run the statevector benchmark for that method first, or use '
            f'--{method}-json /path/to/your/result.json'
        )
    return path


def load_ansatz_data(source, method, code, pool, hf_bits, n_cs):
    with open(source, encoding='utf-8') as f:
        result = json.load(f)
    if result.get('mode') != method or result.get('backend') != 'sv':
        raise ValueError(f'{source} is not an ideal-statevector {method} result')
    if not np.isclose(float(result.get('free_param_fraction')), 0.25):
        raise ValueError(f'{source} is not a 25% operator-retention result')
    selected = list(map(int, result['selected_indices']))
    params = np.asarray(result['optimal_parameters'], dtype=float)
    if len(selected) != 4 or params.shape != (4,):
        raise ValueError(f'{method}: expected four selected operators and four angles')
    if len(set(selected)) != 4 or any(i < 0 or i >= len(pool) for i in selected):
        raise ValueError(f'{method}: invalid selected operator indices')
    ansatz = code._prepare_cs_ansatz(hf_bits, n_cs, pool, selected)
    return result, ansatz, params


def main():
    parser = argparse.ArgumentParser(description='Fixed-parameter QFIM vs gradient hardware comparison')
    parser.add_argument('--runtime-backend', default='ibm_kingston')
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--qfim-json', type=Path)
    parser.add_argument('--gradient-json', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Only validate saved circuits locally; no IBM jobs')
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error('--shots must be positive')

    code = load_original_code()
    filename = 'HCl_STO-3G_SINGLET_JW'
    H, pool, hf_bits, n_cs, _, _ = code._build_cs_hamiltonian(filename)
    cases = {}
    for method in ('qfim', 'gradient'):
        source = getattr(args, f'{method}_json') or load_saved_result(code.DATA_DIR, method)
        saved, ansatz, params = load_ansatz_data(source, method, code, pool, hf_bits, n_cs)
        exact_local_energy = float(np.real(Statevector.from_instruction(
            ansatz.assign_parameters(params)
        ).expectation_value(H)))
        expected = float(saved['energy'])
        delta = abs(exact_local_energy - expected)
        print(f'{method}: selected={saved["selected_indices"]}, '
              f'ideal check={exact_local_energy:.9f} Ha, '
              f'saved={expected:.9f} Ha, difference={delta:.3g} Ha')
        if delta > 1e-5:
            raise ValueError(f'{method}: ideal statevector verification failed; do not submit hardware jobs')
        cases[method] = dict(ansatz=ansatz, params=params, source=str(source),
                             ideal_energy=exact_local_energy,
                             selected_indices=saved['selected_indices'])

    if args.check_only:
        print('Local verification passed. No hardware jobs were submitted.')
        return

    service = QiskitRuntimeService()
    backend = service.backend(args.runtime_backend)
    estimator = EstimatorV2(mode=backend, options={'default_shots': args.shots})
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3)
    outputs = []
    for method in ('qfim', 'gradient'):
        item = cases[method]
        isa_ansatz = pass_manager.run(item['ansatz'])
        isa_H = H.apply_layout(isa_ansatz.layout)
        print(f'Submitting {method} fixed-angle energy measurement to {backend.name}...')
        job = estimator.run([(isa_ansatz, isa_H, item['params'])], precision=None)
        pub = job.result()[0]
        energy = float(np.asarray(pub.data.evs).item())
        stds = getattr(pub.data, 'stds', None)
        std = float(np.asarray(stds).item()) if stds is not None else None
        output = {
            'method': method, 'backend': backend.name,
            'shots_requested': args.shots,
            'job_id': job.job_id(),
            'selected_indices': item['selected_indices'],
            'optimal_parameters': item['params'].tolist(),
            'saved_statevector_json': item['source'],
            'ideal_statevector_energy': item['ideal_energy'],
            'hardware_energy': energy,
            'hardware_estimated_std': std,
            'fci_error_mha': 1000 * abs(energy - (-455.15706679314883)),
            'transpiled_depth': isa_ansatz.depth(),
        }
        outputs.append(output)
        print(f'{method}: hardware={energy:.9f} Ha, FCI error={output["fci_error_mha"]:.3f} mHa')
        outdir = ROOT / 'fixed_hardware_results'
        outdir.mkdir(exist_ok=True)
        path = outdir / f'{method}_{backend.name}_{args.shots}shots.json'
        path.write_text(json.dumps(output, indent=2), encoding='utf-8')
        print(f'Saved {path}')

    print('\nHardware energies (smaller FCI error is better):')
    for o in outputs:
        print(f'{o["method"]}: {o["fci_error_mha"]:.3f} mHa')
    print('One measurement per ansatz is not enough to establish statistical significance.')


if __name__ == '__main__':
    main()
