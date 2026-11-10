import argparse
import json
import os

from pyscf import gto
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.units import DistanceUnit
from pathlib import Path

DATA_DIR = str(Path(__file__).resolve().parent / "data")


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", type=str, default="Be")
    parser.add_argument("-b", "--backend", type=str, default="stn")  # allow simulator or runtime selection
    parser.add_argument("--runtime-backend", type=str, default=None, help="IBM Runtime backend when --backend runtime.")  # configure target quantum hardware
    parser.add_argument("--runtime-shots", type=int, default=4000, help="Shot count for runtime estimator evaluations.")  # control energy evaluation fidelity
    parser.add_argument("-i", "--iterations", type=int, default=1000)
    parser.add_argument("-f", "--free-param-fraction", type=float, default=1)
    parser.add_argument("-m", "--mode", type=str, default="original")
    return parser


def read_symmer_data(filename: str) -> dict:
    with open(
        f"{DATA_DIR}/hamiltonian_data/{filename}.json", "r", encoding="utf-8"
    ) as infile:
        data_dict = json.load(infile)["data"]
    data_dict["geometry"] = "\n".join(data_dict["geometry"].split("\n")[2:])
    data_dict["atoms"] = gto.format_atom(data_dict["geometry"], unit=data_dict["unit"])
    return data_dict


def get_mole(data_dict: dict) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = data_dict["geometry"]
    mol.basis = data_dict["basis"]
    mol.charge = data_dict["charge"]
    mol.spin = data_dict["spin"]
    mol.unit = data_dict["unit"]
    mol.build()
    return mol


def get_energies(data_dict) -> tuple[float, float, float]:
    return (
        data_dict["calculated_properties"]["HF"]["energy"],
        data_dict["calculated_properties"]["FCI"]["energy"],
        get_mole(data_dict).energy_nuc(),
    )


def get_problem(data_dict) -> BaseProblem:
    molecule_info = MoleculeInfo(
        symbols=[atom[0] for atom in data_dict["atoms"]],
        coords=[atom[1] for atom in data_dict["atoms"]],
        multiplicity=2 * data_dict["spin"] + 1,
        charge=data_dict["charge"],
        units=DistanceUnit.ANGSTROM if data_dict["unit"] == "A" else DistanceUnit.BOHR,
    )
    driver = PySCFDriver.from_molecule(molecule_info, basis=data_dict["basis"])
    return driver.run()
