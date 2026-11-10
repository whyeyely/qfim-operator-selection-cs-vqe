'''
CS-VQE-SQD: QFIM-based operator selection

This is our extension of the CS-VQE-SQD code! The main thing we're testing is
whether selecting excitation operators using the QFIM gives better VQE
accuracy than gradient-based pruning or random selection, with the same
number of parameters.

Change MODE and KEEP_FRACTION below to run the different comparisons.
'original' keeps all operators. 'sv' runs the statevector simulation;
'runtime' uses IBM hardware. The CS-only SQD option diagonalizes sampled
computational subspaces, not the full fermionic SQD calculation.

Here's a set of example parameters for HCl, QFIM selection, 25% operator retention:

    -t HCl -b sv -m qfim -f 0.25 -i 500 --sqd-path cs --sqd-shots 10000 --sqd-iterations 3 --sqd-batches 3 --sqd-samples-per-batch 8

Change -m to gradient or random for the corresponding baselines.
For random selection, use --selection-seed to specify the trial seed.
'''

# ---------------- EXPERIMENT SETTINGS ----------------
TARGET = 'HCl'
BACKEND = 'sv'                 # 'sv', 'stn', or 'runtime'
MODE = 'qfim'                 # choose 'original', 'gradient', 'qfim', or 'random'
KEEP_FRACTION = 0.25
MAX_ITERATIONS = 500
SELECTION_SEED = 42           # random subset seed; also QFIM sampling seed

QFIM_POINTS = 8
QFIM_REGULARIZATION = 1e-8

SQD_PATH = 'cs'               # 'cs' or 'fermion'
SQD_SHOTS = 10000
SQD_ITERATIONS = 3
SQD_BATCHES = 3
SQD_SAMPLES_PER_BATCH = 8
SQD_SEED = 42

RUNTIME_BACKEND = None        # set an IBM backend name for BACKEND='runtime'
RUNTIME_SHOTS = 4000
# -----------------------------------------------------

import json
import math
import os
import pickle
import time
from tqdm import tqdm
import warnings
from typing import List, Sequence

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.subsampling import subsample
from qiskit import transpile
from qiskit_algorithms.gradients import QFI, ReverseQGT

from symmer.operators import PauliwordOp, QuantumState
from symmer.projection import ContextualSubspace, QubitTapering

from pyscf import ao2mo, mcscf, scf

from utils import (
    DATA_DIR,
    get_arg_parser,
    get_energies,
    get_mole,
    read_symmer_data,
)

CS_GRADIENT_DIR = f"{DATA_DIR}/gradients_cs_single_term_1028"
CS_OPERATOR_DIR = f"{DATA_DIR}/cs_op_11051"
CS_SQD_DIR = os.path.join(DATA_DIR, "cs_vqe_sqd")


def _load_cs_gradient_data(filename: str) -> dict:
    gradient_path = os.path.join(CS_GRADIENT_DIR, f"{filename}.json")
    if not os.path.exists(gradient_path):
        raise FileNotFoundError(
            f"Gradient data not found at {gradient_path}. "
            "Please run get_cs_vqe_gradient_single_term first."
        )
    with open(gradient_path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def _bits_to_int(bits: Sequence[int]) -> int:
    """Interpret a bit sequence (most-significant bit first) as an integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _deserialize_lookup(data: dict) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cs_bits_str, entry in data.items():
        cs_bits = np.array([int(ch) for ch in cs_bits_str], dtype=int)
        jw_bits = np.array(
            [[int(ch) for ch in jw_str] for jw_str in entry["jw_bitstrings"]], dtype=int
        )
        weights = np.array(entry["weights"], dtype=float)
        lookup[_bits_to_int(cs_bits)] = (jw_bits, weights)
    return lookup


def _build_cs_to_jw_lookup(
    taper: QubitTapering,
    cs_projector: ContextualSubspace,
    n_full_qubits: int,
    n_cs_qubits: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Accumulate weights linking contextual bitstrings to JW bitstrings."""
    tolerance = 1e-9
    lookup: dict[int, dict[tuple[int, ...], float]] = {}

    for index in tqdm(range(1 << n_full_qubits), desc="CS→JW map"):
        bits_full = np.array(
            [int(ch) for ch in f"{index:0{n_full_qubits}b}"], dtype=int
        )
        state_full = QuantumState(
            bits_full.reshape(1, -1),
            np.array([1.0], dtype=complex),
        )

        tapered_state = taper.project_state(state_to_project=state_full)
        coeffs_tapered = np.asarray(tapered_state.state_op.coeff_vec)
        if coeffs_tapered.size == 0 or np.all(np.abs(coeffs_tapered) < tolerance):
            continue

        cs_state = cs_projector.project_state(state_to_project=tapered_state)
        coeffs_cs = np.asarray(cs_state.state_op.coeff_vec)
        if coeffs_cs.size == 0:
            continue

        for basis_bits, coeff in zip(cs_state.state_matrix, coeffs_cs, strict=False):
            weight = float(abs(coeff) ** 2)
            if weight < tolerance:
                continue
            cs_int = _bits_to_int(basis_bits)
            key = tuple(int(b) for b in bits_full)
            bucket = lookup.setdefault(cs_int, {})
            bucket[key] = bucket.get(key, 0.0) + weight

    lookup_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cs_int, weight_dict in lookup.items():
        jw_bits = np.array(list(weight_dict.keys()), dtype=int)
        weights = np.array(list(weight_dict.values()), dtype=float)
        total = weights.sum()
        if total > 0:
            weights /= total
        lookup_arrays[cs_int] = (jw_bits, weights)

    return lookup_arrays


def _lift_cs_bitstrings(
    bitstring_matrix_cs: np.ndarray,
    probs_array_cs: np.ndarray,
    lookup_map: dict[int, tuple[np.ndarray, np.ndarray]],
    num_elec_a: int,
    num_elec_b: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Distribute contextual probabilities across JW bitstrings."""
    if bitstring_matrix_cs.size == 0:
        return np.empty((0, 0), dtype=bool), np.empty(0), 0.0

    prob_accumulator: dict[tuple[int, ...], float] = {}
    captured_prob = 0.0

    for cs_bits, prob in zip(bitstring_matrix_cs, probs_array_cs, strict=False):
        cs_int = _bits_to_int(cs_bits)
        entry = lookup_map.get(cs_int)
        if entry is None:
            continue
        jw_candidates, weights = entry
        if jw_candidates.size == 0:
            continue
        weights = np.asarray(weights, dtype=float)
        if weights.sum() == 0:
            weights = np.ones_like(weights)

        if num_elec_a >= 0 and num_elec_b >= 0:
            half = jw_candidates.shape[1] // 2
            left = jw_candidates[:, :half]
            right = jw_candidates[:, half:]
            mask = (left.sum(axis=1) == num_elec_b) & (right.sum(axis=1) == num_elec_a)
            filtered = jw_candidates[mask]
            filtered_weights = weights[mask]
        else:
            filtered = jw_candidates
            filtered_weights = weights

        if filtered.size == 0:
            filtered = jw_candidates
            filtered_weights = weights

        weight_sum = filtered_weights.sum()
        if weight_sum <= 0:
            filtered_weights = np.ones_like(filtered_weights)
            weight_sum = filtered_weights.sum()

        for bits, w in zip(filtered, filtered_weights, strict=False):
            key = tuple(int(b) for b in bits)
            contribution = prob * (w / weight_sum)
            prob_accumulator[key] = prob_accumulator.get(key, 0.0) + contribution
            captured_prob += contribution

    if not prob_accumulator:
        return np.empty((0, 0), dtype=bool), np.empty(0), captured_prob

    bitstrings = np.array(list(prob_accumulator.keys()), dtype=int)
    probs = np.array(list(prob_accumulator.values()), dtype=float)
    total = probs.sum()
    if total > 0:
        probs /= total
    if captured_prob < 1 - 1e-9:
        print(
            f"Warning: only {captured_prob:.6f} of contextual probability mass mapped to JW states."
        )
    return bitstrings.astype(bool), probs, captured_prob


def _build_cs_hamiltonian(filename: str):
    with open(os.path.join(CS_OPERATOR_DIR, f"{filename}.pckl"), "rb") as infile:
        cs_data = pickle.load(infile)

    h_cs_dict = cs_data["H_cs"]
    UCC_taper = PauliwordOp.from_dictionary(cs_data["UCC_taper"])
    n_cs_qubits = cs_data["n_cs"]

    H_cs_symmer = PauliwordOp.from_dictionary(h_cs_dict)
    pauli_list: List[str] = []
    coeffs: List[complex] = []
    for pauli_str, coeff in H_cs_symmer.to_dictionary.items():
        pauli_list.append(pauli_str)
        coeffs.append(coeff)
    H_qiskit = SparsePauliOp(pauli_list, coeffs)

    with open(
        os.path.join(DATA_DIR, "hamiltonian_data", f"{filename}.json"),
        "r",
        encoding="utf-8",
    ) as infile:
        data_dict = json.load(infile)

    hf_bitstring = cs_data.get("cs_hf_bitstring")
    print(f"Loaded CS Hartree-Fock bitstring: {hf_bitstring}")
    single_term_excitation_pool: List[SparsePauliOp] = []
    single_term_data = cs_data.get("single_term_excitation_pool")
    print(f"Single-term excitation data: {single_term_data}")
    if single_term_data:
        print(f"Loading {len(single_term_data)} single-term excitation operators.")
        for item in single_term_data:
            coeff_real, coeff_imag = item["coeff"]
            single_term_excitation_pool.append(
                SparsePauliOp([item["pauli"]], [complex(coeff_real, coeff_imag)])
            )

    lookup_data = cs_data.get("cs_to_jw_lookup")
    print(f"CS to JW lookup data: {lookup_data}")
    cs_to_jw_lookup = _deserialize_lookup(lookup_data) if lookup_data else {}

    needs_projector = (
        hf_bitstring is None
        or not single_term_excitation_pool
        or not cs_to_jw_lookup
    )
    print(f"CS Hamiltonian requires projector setup: {needs_projector}")
    if needs_projector:
        H_full = PauliwordOp.from_dictionary(data_dict["hamiltonian"])
        QT = QubitTapering(H_full)
        hf_state = QuantumState(np.asarray(data_dict["data"]["hf_array"]))
        H_taper = QT.taper_it(ref_state=hf_state)
        cs_projector = ContextualSubspace(
            H_taper,
            noncontextual_strategy="SingleSweep_magnitude",
            unitary_partitioning_method="LCU",
        )
        cs_projector.update_stabilizers(
            n_qubits=n_cs_qubits,
            strategy="aux_preserving",
            aux_operator=UCC_taper,
        )

        if not single_term_excitation_pool:
            print("Building single_term_excitation_pool")
            UCC_cs = cs_projector.project_onto_subspace(operator_to_project=UCC_taper)
            identity_pauli = "I" * n_cs_qubits
            for op in tqdm(UCC_cs):
                op_dict = op.to_dictionary
                best_term = None
                best_coeff = None
                for pauli_string, coeff in op_dict.items():
                    coeff = complex(coeff)
                    if pauli_string == identity_pauli or np.isclose(abs(coeff), 0.0):
                        continue
                    if best_term is None or abs(coeff) > abs(best_coeff):
                        best_term = pauli_string
                        best_coeff = coeff
                if best_term is None:
                    continue
                single_term_excitation_pool.append(
                    SparsePauliOp([best_term], [best_coeff])
                )

        if hf_bitstring is None:
            print("Building hf_bitstring")
            tapered_hf_state = QT.tapered_ref_state
            cs_hf_state = cs_projector.project_state(tapered_hf_state).normalize
            if cs_hf_state.state_op.n_terms != 1:
                raise ValueError(
                    "Projected Hartree-Fock state is not a computational basis determinant."
                )
            hf_bitstring = cs_hf_state.state_matrix[0].tolist()

        if not cs_to_jw_lookup:
            print("Building cs_to_jw_lookup")
            cs_to_jw_lookup = _build_cs_to_jw_lookup(
                QT, cs_projector, H_full.n_qubits, n_cs_qubits
            )
    else:
        H_full = PauliwordOp.from_dictionary(cs_data["H"])

    return (
        H_qiskit,
        single_term_excitation_pool,
        hf_bitstring if isinstance(hf_bitstring, list) else list(hf_bitstring),
        n_cs_qubits,
        cs_to_jw_lookup,
        H_full.n_qubits,
    )

# IBM Runtime job mode, compatible with accounts without session access.
def _build_runtime_primitives(runtime_backend, runtime_shots, sampler_shots):
    if not runtime_backend:
        raise ValueError('Set RUNTIME_BACKEND or --runtime-backend to a QPU name.')
    service = QiskitRuntimeService()
    backend = service.backend(runtime_backend)
    estimator = EstimatorV2(mode=backend, options={'default_shots': runtime_shots})
    sampler = SamplerV2(mode=backend, options={'default_shots': sampler_shots})
    return estimator, sampler, backend


def select_random_operators(total_excitations, n_keep, seed):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total_excitations, size=n_keep, replace=False).tolist())


def _select_excitation_indices(
    grad_argsort: List[int], total_excitations: int, max_parameters: int
) -> List[int]:
    if max_parameters >= total_excitations:
        return list(range(total_excitations))
    if max_parameters <= 0:
        raise ValueError("max_parameters must be positive when selecting excitations.")
    keep_indices = set(grad_argsort[-max_parameters:])
    return [idx for idx in range(total_excitations) if idx in keep_indices]


def _prepare_cs_ansatz(
    hf_bitstring, n_qubits: int, excitations: List[SparsePauliOp], keep_indices: List[int]
) -> QuantumCircuit:
    ansatz = QuantumCircuit(n_qubits)
    for idx, bit in enumerate(hf_bitstring):
        if bit == 1:
            qubit = n_qubits - idx - 1
            ansatz.x(qubit)

    params = ParameterVector("theta", length=len(keep_indices))
    for param, op in zip(params, (excitations[i] for i in keep_indices)):
        ansatz.append(PauliEvolutionGate(op, time=param), range(n_qubits))
    return ansatz


def _bitstrings_from_vqe_ansatz(
    ansatz: QuantumCircuit,
    optimal_parameters: Sequence[float],
    shots: int,
    sampler: SamplerV2 | None = None,
    backend=None,
) -> tuple[np.ndarray, np.ndarray]:  # optionally use IBM Runtime sampler
    """Sample bitstrings from the optimised CS-VQE ansatz for SQD."""
    if shots <= 0:
        raise ValueError("shots must be positive")

    if len(optimal_parameters) == 0:
        bound_ansatz = ansatz
    else:
        bound_ansatz = ansatz.assign_parameters(optimal_parameters)

    if sampler is None:  # default to noiseless statevector sampling
        state = Statevector.from_instruction(bound_ansatz)  # build exact state from ansatz
        state.seed(42) #added seed for reproducibiity
        counts = state.sample_counts(shots=shots)  # draw synthetic samples from statevector
    else:  # runtime sampler requested
        sample_circuit = bound_ansatz.copy()  # avoid mutating the ansatz used elsewhere
        if sample_circuit.num_clbits == 0:
            sample_circuit.measure_all()  # ensure sampler outputs measurement data
        sample_circuit = _to_isa(sample_circuit, backend, 1)
        job = sampler.run([sample_circuit], shots=shots)
        result = job.result()  # block until sampler job completes
        pub_result = result[0]  # Sampler V2 returns a PrimitiveResult of SamplerPubResults
        bit_array = pub_result.join_data()  # consolidate all classical registers
        counts = bit_array.get_counts()  # recover bitstring counts directly from shots
        if not counts:
            raise RuntimeError(
                "Sampler returned no measurement data; ensure the circuit contains measurements."
            )
    return counts_to_arrays(counts)


def _build_pyscf_data(data_dict: dict) -> dict:
    """Reproduce the notebook's PySCF workflow to obtain integrals for SQD."""
    mol = get_mole(data_dict)
    multiplicity = 2 * data_dict["spin"] + 1
    open_shell = multiplicity != 1

    rhf_solver = scf.RHF(mol)
    hf_energy = rhf_solver.kernel()

    n_frozen = 0
    nao = mol.nao_nr()
    active_space = range(n_frozen, nao)
    num_orbitals = len(active_space)
    nelec_a, nelec_b = mol.nelec

    cas = mcscf.CASCI(rhf_solver, num_orbitals, (nelec_a, nelec_b))
    mo = cas.sort_mo(active_space, base=0)
    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)

    exact_energy = cas.run().e_tot

    return {
        "hf_energy": hf_energy,
        "exact_energy": exact_energy,
        "nuclear_repulsion_energy": nuclear_repulsion_energy,
        "hcore": hcore,
        "eri": eri,
        "open_shell": open_shell,
        "spin_sq": 0,
        "num_orbitals": num_orbitals,
        "num_elec_a": nelec_a,
        "num_elec_b": nelec_b,
    }


def _solve_small_cs_subspace(bitstrings: np.ndarray, hamiltonian: SparsePauliOp) -> float:
    """Diagonalize H projected onto the unique sampled computational basis states.

    The contextual HCl space has only three qubits, so a dense eigensolver
    also handles one-dimensional batches, unlike sparse ARPACK.
    Bitstring columns are most-significant-bit first, as in Qiskit counts.
    """
    bits = np.asarray(bitstrings, dtype=np.int8)
    if bits.ndim != 2 or bits.shape[1] != hamiltonian.num_qubits:
        raise ValueError("Sampled bitstrings have the wrong shape for the CS Hamiltonian")
    if bits.shape[0] == 0:
        raise ValueError("Cannot diagonalize an empty CS subspace")
    indices = np.unique(bits @ (1 << np.arange(bits.shape[1] - 1, -1, -1)))
    dense_h = hamiltonian.to_matrix()
    sub_h = dense_h[np.ix_(indices, indices)]
    sub_h = (sub_h + sub_h.conj().T) / 2  # suppress numerical anti-Hermitian noise
    return float(np.linalg.eigvalsh(sub_h)[0])


def _array_with_none(array: np.ndarray) -> list:
    """Convert a NumPy array to a JSON-serialisable list with NaNs -> None."""
    if array.ndim == 1:
        return [
            (None if not math.isfinite(float(val)) else float(val))
            for val in array
        ]
    return [
        [
            (None if not math.isfinite(float(val)) else float(val))
            for val in row
        ]
        for row in array
    ]


def _occupancy_history_to_list(history: list[tuple[np.ndarray, np.ndarray]]) -> list:
    result: list = []
    for occ in history:
        if occ is None:
            result.append(None)
            continue
        alpha, beta = occ
        result.append([list(np.asarray(alpha, dtype=float)), list(np.asarray(beta, dtype=float))])
    return result

def _to_isa(circ, backend, opt_level=1):
    # Transpile parameters & PauliEvolutionGate into backend ISA (basis gates, coupling)
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    return pm.run(circ)

def circuit_resources(ansatz):
    compiled = transpile(
        ansatz,
        basis_gates=["rz", "sx", "x", "cx"], #standardizing gate set
        optimization_level=3,
        seed_transpiler=42,
    )

    counts = compiled.count_ops()

    return {
        "qubits": compiled.num_qubits,
        "parameters": ansatz.num_parameters,
        "depth": compiled.depth(),
        "cx_count": int(counts.get("cx", 0)),
        "total_gates": int(sum(counts.values())),
    }

#Yi --- compute QFIM w reverse quantum geometric tensor
def compute_qfim(ansatz, parameters):
    qfi = QFI(ReverseQGT())
    result = qfi.run(
        [ansatz.decompose(reps=3)],
        [list(parameters)]
    ).result()
    return np.asarray(result.qfis[0], dtype=float)

#Yi --- select excitation operators at multiple param pts
def select_qfim_operators(ansatz, n_keep, n_points=8, seed=42, regularization=1e-8):
    rng = np.random.default_rng(seed)
    n_parameters = ansatz.num_parameters

    points = [np.zeros(n_parameters)]
    points += [
        rng.normal(0, 0.5, n_parameters)
        for _ in range(n_points - 1)
    ]

    F_total = np.zeros((n_parameters, n_parameters))

    for theta in points:
        F_total += compute_qfim(ansatz, theta)

    F_total /= len(points)
    F_total = (F_total + F_total.T) / 2

    selected = []
    remaining = list(range(n_parameters))
    for _ in range(n_keep):
        best_index = None
        best_score = -np.inf

        for i in remaining:
            trial = selected + [i]
            submatrix = F_total[np.ix_(trial, trial)]
            M = submatrix + regularization * np.eye(len(trial))

            sign, logdet = np.linalg.slogdet(M)

            if sign > 0 and logdet > best_score:
                best_score = logdet
                best_index = i

        selected.append(best_index)
        remaining.remove(best_index)

    return sorted(selected), np.linalg.eigvalsh(F_total)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = get_arg_parser()
    parser.add_argument("--sqd-shots", type=int, default=100_000)
    parser.add_argument("--sqd-iterations", type=int, default=5)
    parser.add_argument("--sqd-batches", type=int, default=5)
    parser.add_argument("--sqd-samples-per-batch", type=int, default=20)
    parser.add_argument("--sqd-max-davidson-cycles", type=int, default=2000)
    parser.add_argument("--sqd-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="If set, save SQD diagnostic plots to the given filepath.",
    )
    parser.add_argument(
        "--sqd-path",
        type=str,
        default="fermion",
        choices=["fermion", "cs"],
        help="SQD diagonalization path: 'fermion' (JW, default) or 'cs' (CS-only/qubit).",
    )
    parser.add_argument(
        "--stn-max-nonclifford",
        type=int,
        default=None,
        help="Number of ansatz parameters kept free (non-Clifford) during STN "
             "evaluation. The rest are rounded to the nearest Clifford angle "
             "(k*pi/2). None (default) disables rounding.",
    )
    parser.add_argument(
        "--stn-rounding-strategy",
        type=str,
        default="min_diff",
        choices=["min_diff", "last"],
        help="Strategy for choosing which parameters to round to Clifford angles. "
             "'min_diff' rounds those already closest to a Clifford angle; "
             "'last' rounds the last (N - max_nonclifford) parameters.",
    )

    # Defaults may still be overridden from PyCharm or the command line.
    parser.set_defaults(
        target=TARGET,
        backend=BACKEND,
        mode=MODE,
        free_param_fraction=KEEP_FRACTION,
        iterations=MAX_ITERATIONS,
        selection_seed=SELECTION_SEED,
        sqd_path=SQD_PATH,
        sqd_shots=SQD_SHOTS,
        sqd_iterations=SQD_ITERATIONS,
        sqd_batches=SQD_BATCHES,
        sqd_samples_per_batch=SQD_SAMPLES_PER_BATCH,
        sqd_seed=SQD_SEED,
        runtime_backend=RUNTIME_BACKEND,
        runtime_shots=RUNTIME_SHOTS,
    )
    args = parser.parse_args()
    print(args)

    stage_times: dict[str, float] = {}

    filename = f"{args.target}_STO-3G_SINGLET_JW"
    data_dict = read_symmer_data(filename)
    hf_energy, fci_energy, nuclear_repulsion_energy = get_energies(data_dict)
    print(f"{fci_energy=} {hf_energy=} {nuclear_repulsion_energy=}")

    t0 = time.perf_counter()
    if args.sqd_path == "fermion":
        # PySCF CASCI is only needed for fermionic SQD.
        pyscf_data = _build_pyscf_data(data_dict)
    else:
        # The existing Hamiltonian JSON contains the reference energies.
        # Avoid an unnecessary full CASCI calculation for the 3-qubit CS test.
        pyscf_data = {
            "hf_energy": float(hf_energy),
            "exact_energy": float(fci_energy),
            "nuclear_repulsion_energy": float(nuclear_repulsion_energy),
        }

    (
        H_qiskit,
        excitation_pool,
        hf_bitstring,
        n_cs_qubits,
        cs_to_jw_lookup,
        n_full_qubits,
    ) = _build_cs_hamiltonian(filename)
    stage_times["cs_preprocessing"] = time.perf_counter() - t0
    print(f"Exact contextual-subspace ground energy: {np.linalg.eigvalsh(H_qiskit.to_matrix())[0]:.9f} Ha")
    print("Note: H_cs includes nuclear repulsion; it is NOT added again for CS calculations.")

    total_excitations = len(excitation_pool)
    if total_excitations == 0:
        raise ValueError("No excitation operators available in contextual subspace.")

    t0 = time.perf_counter()

    # Yi --- initialize QFIM diagnostics
    qfim_eigenvalues = None

    if args.mode == "gradient":
        gradient_data = _load_cs_gradient_data(filename)
        grad_argsort = gradient_data["grad_argsort"]
        max_parameters = int(total_excitations * args.free_param_fraction)
        max_parameters = min(max_parameters, total_excitations)
        max_parameters = max(max_parameters, 1)
        selected_indices = _select_excitation_indices(
            grad_argsort, total_excitations, max_parameters
        )
        print("Gradient-selected operators:", selected_indices)

    # Yi --- use geometric operator selection instead of gradient pruning
    elif args.mode == "qfim":
        max_parameters = max(
            1, min(int(total_excitations * args.free_param_fraction), total_excitations)
        )

        full_ansatz = _prepare_cs_ansatz(
            hf_bitstring,
            n_cs_qubits,
            excitation_pool,
            list(range(total_excitations)),
        )

        selected_indices, qfim_eigenvalues = select_qfim_operators(
            full_ansatz,
            n_keep=max_parameters,
            n_points=QFIM_POINTS,
            seed=args.selection_seed,
            regularization=QFIM_REGULARIZATION,
        )

        print("QFIM-selected operators:", selected_indices)
        print("QFIM eigenvalues:", qfim_eigenvalues)

    elif args.mode == "random":
        max_parameters = max(1, min(int(total_excitations * args.free_param_fraction), total_excitations))
        selected_indices = select_random_operators(
            total_excitations, max_parameters, args.selection_seed
        )
        print("Random-selected operators:", selected_indices)
    else:
        selected_indices = list(range(total_excitations))
        max_parameters = total_excitations

    ansatz = _prepare_cs_ansatz(hf_bitstring, n_cs_qubits, excitation_pool, selected_indices)

    resources = circuit_resources(ansatz)

    print("\n--- Circuit resources ---")
    print(f"Qubits: {resources['qubits']}")
    print(f"Parameters: {resources['parameters']}")
    print(f"Transpiled circuit depth: {resources['depth']}")
    print(f"CX gates: {resources['cx_count']}")
    print(f"Total gates: {resources['total_gates']}")

    stage_times["ansatz_construction"] = time.perf_counter() - t0

    num_parameters_left = len(selected_indices)

    runtime_sampler = None  # sampler reused for SQD sampling
    runtime_backend_obj = None
    vqe_hamiltonian = H_qiskit
    if args.backend == "sv":
        estimator = StatevectorEstimator(default_precision=0.0)
    elif args.backend == "stn":
        # Legacy STN estimator uses Qiskit V1 primitives; only import on demand.
        from stn_estimator import STNEstimator
        stn_options: dict = {"shots": None}
        if args.stn_max_nonclifford is not None:
            stn_options["max_parameters"] = args.stn_max_nonclifford
            stn_options["parameters_to_round"] = args.stn_rounding_strategy
            print(
                f"Clifford rounding enabled: keeping {args.stn_max_nonclifford} "
                f"free parameters, strategy={args.stn_rounding_strategy}"
            )
        estimator = STNEstimator(options=stn_options)
    elif args.backend == "runtime":
        (
            estimator,
            runtime_sampler,
            runtime_backend_obj,
        ) = _build_runtime_primitives(
            args.runtime_backend, args.runtime_shots, args.sqd_shots
        )
        ansatz = _to_isa(ansatz, runtime_backend_obj, 3)
        # Observable must follow the physical-qubit mapping of the compiled ansatz.
        vqe_hamiltonian = H_qiskit.apply_layout(ansatz.layout)
    else:
        raise ValueError(f"Invalid backend {args.backend}")

    energy_history = []

    def callback(iteration, parameters, energy, metadata):
        total_energy = float(np.real(energy))  # H_cs already includes nuclear repulsion
        energy_history.append(total_energy)
        print(iteration, total_energy, parameters[:5])

    vqe = VQE(
        estimator,
        ansatz,
        COBYLA(maxiter=args.iterations, rhobeg=0.3),
        initial_point=np.zeros(ansatz.num_parameters),
        callback=callback,
    )

    t0 = time.perf_counter()
    result = vqe.compute_minimum_eigenvalue(vqe_hamiltonian)
    stage_times["vqe_optimization"] = time.perf_counter() - t0
    print(result)
    print("Estimator:", type(estimator).__name__)

    if result.optimal_point is None:
        optimal_parameters = np.zeros(ansatz.num_parameters)
    elif isinstance(result.optimal_point, np.ndarray):
        optimal_parameters = result.optimal_point
    else:
        optimal_parameters = np.asarray(result.optimal_point, dtype=float)

    t0 = time.perf_counter()
    sqd_shots = args.sqd_shots
    bitstring_matrix_cs, probs_array_cs = _bitstrings_from_vqe_ansatz(
        ansatz,  # sample from optimised ansatz
        optimal_parameters,  # plug in VQE optimum parameters
        sqd_shots,  # default sample count for simulators
        sampler=runtime_sampler,
        backend=runtime_backend_obj,
    )
    bitstring_matrix_cs = bitstring_matrix_cs.astype(int, copy=False)
    stage_times["sampling"] = time.perf_counter() - t0


    if args.sqd_path == "fermion":
        # JAX-dependent SQD modules are only needed for this path.
        from qiskit_addon_sqd.configuration_recovery import recover_configurations
        from qiskit_addon_sqd.subsampling import postselect_and_subsample
        from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs, solve_fermion
        # (A) original behavior: lift CS samples to JW & continue as before
        t0 = time.perf_counter()
        bitstring_matrix_full, probs_array_full, captured_prob = _lift_cs_bitstrings(
            bitstring_matrix_cs,
            probs_array_cs,
            cs_to_jw_lookup,
            pyscf_data["num_elec_a"],
            pyscf_data["num_elec_b"],
        )
        print(
            f"Sampled {bitstring_matrix_cs.shape[0]} contextual configurations of width "
            f"{n_cs_qubits} and distributed them across {bitstring_matrix_full.shape[0]} "
            f"JW configurations of width {n_full_qubits} with {sqd_shots} shots "
            f"(captured probability {captured_prob:.6f})."
        )
        if bitstring_matrix_full.size == 0:
            raise RuntimeError("No JW bitstrings available after lifting CS samples.")

        # ---- One-time reorder from interleaved [α0,β0,α1,β1,...] → [β-block | α-block]
        B = bitstring_matrix_full
        n = B.shape[1] // 2

        # Symmer's JW bitstrings are interleaved: [α0,β0,α1,β1,...,α_{N-1},β_{N-1}].
        # qiskit_addon_sqd expects block order with MSB-first within each half:
        #   [β_{N-1},...,β_0, α_{N-1},...,α_0]
        # so position i in each half maps to orbital (N-1-i) via 2^(N-1-i).
        perm = np.r_[np.arange(1, 2*n, 2)[::-1], np.arange(0, 2*n, 2)[::-1]]
        bitstring_matrix_full = B[:, perm]
        print("Reordered interleaved JW bitstrings to block spin order (MSB-first).")
        stage_times["lifting"] = time.perf_counter() - t0





        # set common aliases so the rest of your code below is unchanged
        bs_source_matrix = bitstring_matrix_full
        bs_source_probs  = probs_array_full
        is_cs_only = False
    else:
        # (B) CS-only path: stay in contextual subspace, no lifting
        print(
            f"CS-only SQD: using {bitstring_matrix_cs.shape[0]} contextual samples "
            f"(width {n_cs_qubits}) from the ansatz directly; no lifting."
        )
        # set common aliases so the rest of your code below is unchanged
        bs_source_matrix = bitstring_matrix_cs
        bs_source_probs  = probs_array_cs
        captured_prob = float(np.sum(probs_array_cs))  # for logging/output symmetry
        is_cs_only = True

    t0_sqd = time.perf_counter()
    rng = np.random.default_rng(args.sqd_seed)
    iterations = args.sqd_iterations
    n_batches = args.sqd_batches
    samples_per_batch = args.sqd_samples_per_batch
    max_davidson_cycles = args.sqd_max_davidson_cycles

    e_hist = np.full((iterations, n_batches), np.nan)
    s_hist = np.full((iterations, n_batches), np.nan)
    occupancy_hist: list[tuple[np.ndarray, np.ndarray]] = []
    avg_occupancy: tuple[np.ndarray, np.ndarray] | None = None

    for i in range(iterations):
        print(f"Starting configuration recovery iteration {i}")

        # Recovery: only do it on JW (fermion) path, since it needs Nα/Nβ
        if not is_cs_only:
            if avg_occupancy is None:
                bs_mat_tmp = bs_source_matrix
                probs_arr_tmp = bs_source_probs
            else:
                seed_recover = int(rng.integers(0, 2**63))
                bs_mat_tmp, probs_arr_tmp = recover_configurations(
                    bs_source_matrix,
                    bs_source_probs,
                    avg_occupancy,
                    pyscf_data["num_elec_a"],
                    pyscf_data["num_elec_b"],
                    rand_seed=seed_recover,
                )
            if bs_mat_tmp.size == 0:
                print("No configurations after recovery/postprocessing; skipping iteration")
                continue

            # Original batching (kept): postselect + subsample in JW
            seed_batches = int(rng.integers(0, 2**63))
            batches = postselect_and_subsample(
                bs_mat_tmp,
                probs_arr_tmp,
                hamming_right=pyscf_data["num_elec_a"],
                hamming_left=pyscf_data["num_elec_b"],
                samples_per_batch=samples_per_batch,
                num_batches=n_batches,
                rand_seed=seed_batches,
            )

        else:
            # CS-only path: subsample batches directly in CS (no postselect-by-Hamming)
            seed_batches = int(rng.integers(0, 2**63))
            # NOTE: use modern subsample API (same knobs) for parity with fermion batching
            batches = subsample(
                bs_source_matrix.astype(bool, copy=False),
                bs_source_probs,
                samples_per_batch=samples_per_batch,
                num_batches=n_batches,
                rand_seed=seed_batches,
            )  # returns a list of 2D boolean arrays.

        e_tmp = np.full(n_batches, np.nan)
        s_tmp = np.full(n_batches, np.nan)
        occs_tmp: list[np.ndarray] = []

        for j in range(n_batches):
            bs = batches[j]
            if getattr(bs, "ndim", 0) != 2 or bs.shape[0] == 0:
                print(f"Skipping batch {j}: empty or 1D batch {getattr(bs, 'shape', None)}")
                continue

            try:
                if not is_cs_only:
                    # (A) original JW/fermion solve
                    if bs.shape[1] % 2 != 0:
                        print(f"Skipping batch {j}: bitstring width {bs.shape[1]} not even")
                        continue
                    strs_a, strs_b = bitstring_matrix_to_ci_strs(bs)
                    print(f"Batch {j} subspace dimension: {len(strs_a) * len(strs_b)}")
                    energy_sci, _, avg_occs, spin = solve_fermion(
                        bs,
                        pyscf_data["hcore"],
                        pyscf_data["eri"],
                        open_shell=pyscf_data["open_shell"],
                        spin_sq=pyscf_data["spin_sq"],
                        max_cycle=max_davidson_cycles,
                    )
                    energy_sci += pyscf_data["nuclear_repulsion_energy"]
                    e_tmp[j] = energy_sci
                    s_tmp[j] = spin
                    occs_tmp.append(avg_occs)

                else:
                    # (B) CS-only/qubit solve on H_qiskit
                    # Dense projected diagonalization handles a one-bitstring subspace.
                    # The supplied H_cs contains total molecular energy already.
                    e_tmp[j] = _solve_small_cs_subspace(bs, H_qiskit)
                    # spin & occupancies are JW concepts; keep NaN / empty so plotting still works

            except Exception as err:
                print(f"Batch {j} failed: {type(err).__name__}: {err}")
                continue

        if not is_cs_only and len(occs_tmp) > 0:
            avg_occupancy = tuple(np.mean(occs_tmp, axis=0))
        elif is_cs_only:
            # no occupancy update in CS-only mode
            pass

        e_hist[i, :] = e_tmp
        s_hist[i, :] = s_tmp
        if (not is_cs_only) and (avg_occupancy is not None):
            occupancy_hist.append(avg_occupancy)

    stage_times["sqd_diagonalization"] = time.perf_counter() - t0_sqd

    min_energy_per_iter: list[float | None] = []
    for row in e_hist:
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            min_energy_per_iter.append(None)
        else:
            min_energy_per_iter.append(float(finite.min()))

    best_energy_candidates = [val for val in min_energy_per_iter if val is not None]
    best_sqd_energy = min(best_energy_candidates) if best_energy_candidates else None
    if best_sqd_energy is not None and math.isfinite(best_sqd_energy):
        best_sqd_error = best_sqd_energy - pyscf_data["exact_energy"]
    else:
        best_sqd_error = None
        if best_sqd_energy is not None and not math.isfinite(best_sqd_energy):
            best_sqd_energy = None

    if best_sqd_error is not None and not math.isfinite(best_sqd_error):
        best_sqd_error = None

    if best_sqd_energy is not None:
        print(
            f"Best SQD energy {best_sqd_energy:.6f} Ha "
            f"(error {best_sqd_error if best_sqd_error is not None else float('nan'):.6f} Ha)"
        )
    else:
        print("SQD did not produce a finite energy estimate.")

    if args.plot_output:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not available; skipping SQD diagnostic plots.")
        else:
            x1 = list(range(iterations))
            min_e = [
                (np.nan if val is None else val) for val in min_energy_per_iter
            ]
            e_diff = [
                np.nan if not np.isfinite(val) else abs(val - pyscf_data["exact_energy"])
                for val in min_e
            ]
            vqe_error = abs(float(result.eigenvalue.real) - pyscf_data["exact_energy"])

            chem_accuracy = 0.0016

            if occupancy_hist:
                final_alpha, final_beta = occupancy_hist[-1]
                y2 = np.asarray(final_alpha, dtype=float) + np.asarray(final_beta, dtype=float)
                x2 = list(range(len(y2)))
            else:
                y2 = np.array([])
                x2 = []

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].plot(x1, e_diff, marker="o", label="energy error")
            axs[0].set_xticks(x1)
            axs[0].set_yscale("log")
            axs[0].set_ylim(bottom=1e-4)
            axs[0].axhline(y=chem_accuracy, color="#BF5700", linestyle="--", label="chemical accuracy")
            axs[0].axhline(y=vqe_error, color="#1f77b4", linestyle="-.", label="initial VQE error")
            axs[0].set_title("Approximated Ground State Energy Error vs SQD Iterations")
            axs[0].set_xlabel("Iteration Index")
            axs[0].set_ylabel("Energy Error (Ha)")
            axs[0].legend()

            axs[1].bar(x2, y2, width=0.8)
            if x2:
                axs[1].set_xticks(x2)
            axs[1].set_title("Avg Occupancy per Spatial Orbital")
            axs[1].set_xlabel("Orbital Index")
            axs[1].set_ylabel("Avg Occupancy")

            fig.tight_layout()
            fig.savefig(args.plot_output, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved SQD diagnostics plot to {args.plot_output}")

    optimizer_time = result.optimizer_time
    if optimizer_time is not None:
        optimizer_time = float(optimizer_time)

    stage_times["total"] = sum(stage_times.values())
    print("\n--- Stage timing breakdown ---")
    for stage, t in stage_times.items():
        print(f"  {stage:25s}  {t:8.2f} s  ({t / stage_times['total'] * 100:5.1f}%)")

    data = {
        "target": args.target,
        "circuit_resources": resources,
        # Yi --- save geometric selection results for comparison with gradient pruning
        "qfim_points": QFIM_POINTS if args.mode == "qfim" else None,
        "qfim_regularization": QFIM_REGULARIZATION if args.mode == "qfim" else None,
        "optimizer_evaluations": (
            int(result.cost_function_evals)
            if result.cost_function_evals is not None else None
        ),
        "qfim_eigenvalues": (
            None if qfim_eigenvalues is None else qfim_eigenvalues.tolist()
        ),
        "geometry": data_dict["geometry"],
        "basis": data_dict["basis"],
        "charge": data_dict["charge"],
        "unit": data_dict["unit"],
        "spin": data_dict["spin"],
        "nuclear_repulsion_energy": float(nuclear_repulsion_energy),
        "hf_energy": float(hf_energy),
        "fci_energy": float(fci_energy),
        "backend": args.backend,
        "iterations": args.iterations,
        "mode": args.mode,
        "selection_seed": args.selection_seed if args.mode == "random" else None,
        "free_param_fraction": args.free_param_fraction,
        "n_cs_qubits": n_cs_qubits,
        "n_full_qubits": n_full_qubits,
        "num_excitations_total": total_excitations,
        "num_parameters_left": num_parameters_left,
        "energy": float(result.eigenvalue.real),
        "time": optimizer_time,
        "stage_times": {k: round(v, 4) for k, v in stage_times.items()},
        "stn_max_nonclifford": args.stn_max_nonclifford,
        "stn_rounding_strategy": args.stn_rounding_strategy if args.stn_max_nonclifford is not None else None,
        "optimal_parameters": optimal_parameters.tolist(),
        "energy_history": [float(val) for val in energy_history],
        "selected_indices": [int(idx) for idx in selected_indices],
        "sqd": {
            "shots": sqd_shots,
            "iterations": iterations,
            "batches": n_batches,
            "samples_per_batch": samples_per_batch,
            "max_davidson_cycles": max_davidson_cycles,
            "seed": args.sqd_seed,
            "bitstring_count": int(bs_source_matrix.shape[0]),
            "bitstring_width": int(bs_source_matrix.shape[1]),
            "bitstring_width_contextual": int(n_cs_qubits),
            "contextual_probability_captured": float(captured_prob),
            "energy_history": _array_with_none(e_hist),
            "spin_history": _array_with_none(s_hist),
            "avg_occupancy_history": _occupancy_history_to_list(occupancy_hist),
            "final_avg_occupancy": (
                None
                if avg_occupancy is None
                else [
                    list(np.asarray(avg_occupancy[0], dtype=float)),
                    list(np.asarray(avg_occupancy[1], dtype=float)),
                ]
            ),
            "min_energy_per_iteration": min_energy_per_iter,
            "best_energy": None if best_sqd_energy is None else float(best_sqd_energy),
            "best_energy_error": None if best_sqd_error is None else float(best_sqd_error),
            "exact_energy": float(pyscf_data["exact_energy"]),
            "hf_energy_pyscf": float(pyscf_data["hf_energy"]),
            "nuclear_repulsion_energy_pyscf": float(pyscf_data["nuclear_repulsion_energy"]),
        },
    }

    cliff_tag = f"-cliff{args.stn_max_nonclifford}" if args.stn_max_nonclifford is not None else ""
    seed_tag = f"-seed{args.selection_seed}" if args.mode == "random" else ""
    mode_tag = (
        f"{args.backend}-{args.mode}-{args.free_param_fraction:.2f}"
        f"{seed_tag}{cliff_tag}-sqd-{args.sqd_path}"
    )
    dirpath = os.path.join(CS_SQD_DIR, mode_tag)
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, f"{filename}.json"), "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)
