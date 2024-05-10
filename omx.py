#!/usr/bin/python3
import numpy as np

# from ..unit import (
#     EnergyConversion,
#     ForceConversion,
#     LengthConversion,
#     PressureConversion,
# )

from abc import ABC

from scipy import constants

AVOGADRO = constants.Avogadro  # Avagadro constant
ELE_CHG = constants.elementary_charge  # Elementary Charge, in C
BOHR = constants.value("atomic unit of length")  # Bohr, in m
HARTREE = constants.value("atomic unit of energy")  # Hartree, in Jole
RYDBERG = constants.Rydberg * constants.h * constants.c  # Rydberg, in Jole

# energy conversions
econvs = {
    "eV": 1.0,
    "hartree": HARTREE / ELE_CHG,
    "kJ_mol": 1 / (ELE_CHG * AVOGADRO / 1000),
    "kcal_mol": 1 / (ELE_CHG * AVOGADRO / 1000 / 4.184),
    "rydberg": RYDBERG / ELE_CHG,
    "J": 1 / ELE_CHG,
    "kJ": 1000 / ELE_CHG,
}

# length conversions
lconvs = {
    "angstrom": 1.0,
    "bohr": BOHR * 1e10,
    "nm": 10.0,
    "m": 1e10,
}


def check_unit(unit):
    if unit not in econvs.keys() and unit not in lconvs.keys():
        try:
            eunit = unit.split("/")[0]
            lunit = unit.split("/")[1]
            if eunit not in econvs.keys():
                raise RuntimeError(f"Invaild unit: {unit}")
            if lunit not in lconvs.keys():
                raise RuntimeError(f"Invalid unit: {unit}")
        except Exception:
            raise RuntimeError(f"Invalid unit: {unit}")


class Conversion(ABC):
    def __init__(self, unitA, unitB, check=True):
        """Parent class for unit conversion.

        Parameters
        ----------
        unitA : str
            unit to be converted
        unitB : str
            unit which unitA is converted to, i.e. `1 unitA = self._value unitB`
        check : bool
            whether to check unit validity

        Examples
        --------
        >>> conv = Conversion("foo", "bar", check=False)
        >>> conv.set_value("10.0")
        >>> print(conv)
        1 foo = 10.0 bar
        >>> conv.value()
        10.0
        """
        if check:
            check_unit(unitA)
            check_unit(unitB)
        self.unitA = unitA
        self.unitB = unitB
        self._value = 0.0

    def value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def __repr__(self):
        return f"1 {self.unitA} = {self._value} {self.unitB}"

    def __str__(self):
        return self.__repr__()


class EnergyConversion(Conversion):
    def __init__(self, unitA, unitB):
        """Class for energy conversion.

        Examples
        --------
        >>> conv = EnergyConversion("eV", "kcal_mol")
        >>> conv.value()
        23.06054783061903
        """
        super().__init__(unitA, unitB)
        self.set_value(econvs[unitA] / econvs[unitB])


class LengthConversion(Conversion):
    def __init__(self, unitA, unitB):
        """Class for length conversion.

        Examples
        --------
        >>> conv = LengthConversion("angstrom", "nm")
        >>> conv.value()
        0.1
        """
        super().__init__(unitA, unitB)
        self.set_value(lconvs[unitA] / lconvs[unitB])


class ForceConversion(Conversion):
    def __init__(self, unitA, unitB):
        """Class for force conversion.

        Parameters
        ----------
        unitA, unitB : str
            in format of "energy_unit/length_unit"

        Examples
        --------
        >>> conv = ForceConversion("kJ_mol/nm", "eV/angstrom")
        >>> conv.value()
        0.0010364269656262175
        """
        super().__init__(unitA, unitB)
        econv = EnergyConversion(unitA.split("/")[0], unitB.split("/")[0]).value()
        lconv = LengthConversion(unitA.split("/")[1], unitB.split("/")[1]).value()
        self.set_value(econv / lconv)


class PressureConversion(Conversion):
    def __init__(self, unitA, unitB):
        """Class for pressure conversion.

        Parameters
        ----------
        unitA, unitB : str
            in format of "energy_unit/length_unit^3", or in `["Pa", "pa", "kPa", "kpa", "bar", "kbar"]`

        Examples
        --------
        >>> conv = PressureConversion("kbar", "eV/angstrom^3")
        >>> conv.value()
        0.0006241509074460763
        """
        super().__init__(unitA, unitB, check=False)
        unitA, factorA = self._convert_unit(unitA)
        unitB, factorB = self._convert_unit(unitB)
        eunitA, lunitA = self._split_unit(unitA)
        eunitB, lunitB = self._split_unit(unitB)
        econv = EnergyConversion(eunitA, eunitB).value() * factorA / factorB
        lconv = LengthConversion(lunitA, lunitB).value()
        self.set_value(econv / lconv**3)

    def _convert_unit(self, unit):
        if unit == "Pa" or unit == "pa":
            return "J/m^3", 1
        elif unit == "kPa" or unit == "kpa":
            return "kJ/m^3", 1
        elif unit == "GPa" or unit == "Gpa":
            return "kJ/m^3", 1e6
        elif unit == "bar":
            return "J/m^3", 1e5
        elif unit == "kbar":
            return "kJ/m^3", 1e5
        else:
            return unit, 1

    def _split_unit(self, unit):
        eunit = unit.split("/")[0]
        lunit = unit.split("/")[1][:-2]
        return eunit, lunit

ry2ev = EnergyConversion("rydberg", "eV").value()
kbar2evperang3 = PressureConversion("kbar", "eV/angstrom^3").value()

length_convert = LengthConversion("bohr", "angstrom").value()
energy_convert = EnergyConversion("hartree", "eV").value()
force_convert = ForceConversion("hartree/bohr", "eV/angstrom").value()

import warnings
from collections import OrderedDict

### iterout.c from OpenMX soure code: column numbers and physical quantities ###
# /* 1: */
# /* 2,3,4: */
# /* 5,6,7: force *
# /* 8: x-component of velocity */
# /* 9: y-component of velocity */
# /* 10: z-component of velocity */
# /* 11: Net charge, electron charge is defined to be negative. */
# /* 12: magnetic moment (muB) */
# /* 13,14: angles of spin */

# 15: scf_convergence_flag (optional)
#
# 1. Move the declaration of `scf_convergence_flag` in `DFT.c` to `openmx_common.h`.
# 2. Add `scf_convergence_flag` output to the end of `iterout.c` where `*.md` is written.
# 3. Recompile OpenMX.


def load_atom(lines):
    atom_names = []
    atom_names_mode = False
    for line in lines:
        if "<Atoms.SpeciesAndCoordinates" in line:
            atom_names_mode = True
        elif "Atoms.SpeciesAndCoordinates>" in line:
            atom_names_mode = False
        elif atom_names_mode:
            parts = line.split()
            atom_names.append(parts[1])
    natoms = len(atom_names)
    atom_names_original = atom_names
    atom_names = list(OrderedDict.fromkeys(set(atom_names)))  # Python>=3.7
    atom_names = sorted(
        atom_names, key=atom_names_original.index
    )  # Unique ordering of atomic species
    ntypes = len(atom_names)
    atom_numbs = [0] * ntypes
    atom_types = []
    atom_types_mode = False
    for line in lines:
        if "<Atoms.SpeciesAndCoordinates" in line:
            atom_types_mode = True
        elif "Atoms.SpeciesAndCoordinates>" in line:
            atom_types_mode = False
        elif atom_types_mode:
            parts = line.split()
            for i, atom_name in enumerate(atom_names):
                if parts[1] == atom_name:
                    atom_numbs[i] += 1
                    atom_types.append(i)
    atom_types = np.array(atom_types)
    return atom_names, atom_types, atom_numbs


def load_cells(lines):
    cell, cells = [], []
    for index, line in enumerate(lines):
        if "Cell_Vectors=" in line:
            parts = line.split()
            if len(parts) == 21:  # MD.Type is NVT_NH
                cell.append([float(parts[12]), float(parts[13]), float(parts[14])])
                cell.append([float(parts[15]), float(parts[16]), float(parts[17])])
                cell.append([float(parts[18]), float(parts[19]), float(parts[20])])
            elif len(parts) == 16:  # MD.Type is Opt
                cell.append([float(parts[7]), float(parts[8]), float(parts[9])])
                cell.append([float(parts[10]), float(parts[11]), float(parts[12])])
                cell.append([float(parts[13]), float(parts[14]), float(parts[15])])
            else:
                raise RuntimeError(
                    "Does the file System.Name.md contain unsupported calculation results?"
                )
            cells.append(cell)
            cell = []
    cells = np.array(cells)
    return cells


# load atom_names, atom_numbs, atom_types, cells
def load_param_file(fname, mdname):
    with open(fname) as dat_file:
        lines = dat_file.readlines()
    atom_names, atom_types, atom_numbs = load_atom(lines)

    with open(mdname) as md_file:
        lines = md_file.readlines()
    cells = load_cells(lines)
    return atom_names, atom_numbs, atom_types, cells


def load_coords(lines, atom_names, natoms):
    cnt = 0
    coord, coords = [], []
    for index, line in enumerate(lines):
        if "time=" in line:
            continue
        for atom_name in atom_names:
            atom_name += " "
            if atom_name in line:
                cnt += 1
                parts = line.split()
                for_line = [float(parts[1]), float(parts[2]), float(parts[3])]
                coord.append(for_line)
                # It may be necessary to recompile OpenMX to make scf convergence determination.
                if len(parts) == 15 and parts[14] == "0":
                    warnings.warn("SCF in System.Name.md has not converged!")
        if cnt == natoms:
            coords.append(coord)
            cnt = 0
            coord = []
    coords = np.array(coords)
    return coords


def load_data(mdname, atom_names, natoms):
    with open(mdname) as md_file:
        lines = md_file.readlines()
    coords = load_coords(lines, atom_names, natoms)
    steps = [str(i) for i in range(1, coords.shape[0] + 1)]
    return coords, steps


def to_system_data(fname, mdname):
    data = {}
    (
        data["atom_names"],
        data["atom_numbs"],
        data["atom_types"],
        data["cells"],
    ) = load_param_file(fname, mdname)
    data["coords"], steps = load_data(
        mdname,
        data["atom_names"],
        np.sum(data["atom_numbs"]),
    )
    data["orig"] = np.zeros(3)
    return data, steps


def load_energy(lines):
    energy = []
    for line in lines:
        if "time=" in line:
            parts = line.split()
            ene_line = float(parts[4])  # Hartree
            energy.append(ene_line)
            continue
    energy = energy_convert * np.array(energy)  # Hartree -> eV
    return energy


def load_force(lines, atom_names, atom_numbs):
    cnt = 0
    field, fields = [], []
    for index, line in enumerate(lines):
        if "time=" in line:
            continue
        for atom_name in atom_names:
            atom_name += " "
            if atom_name in line:
                cnt += 1
                parts = line.split()
                for_line = [float(parts[4]), float(parts[5]), float(parts[6])]
                field.append(for_line)
        if cnt == np.sum(atom_numbs):
            fields.append(field)
            cnt = 0
            field = []
    force = force_convert * np.array(fields)
    return force


# load energy, force
def to_system_label(fname, mdname):
    atom_names, atom_numbs, atom_types, cells = load_param_file(fname, mdname)
    with open(mdname) as md_file:
        lines = md_file.readlines()
    energy = load_energy(lines)
    force = load_force(lines, atom_names, atom_numbs)
    return energy, force


if __name__ == "__main__":
    file_name = "Methane2"
    fname = f"{file_name}.dat"
    mdname = f"{file_name}.md"
    atom_names, atom_numbs, atom_types, cells = load_param_file(fname, mdname)
    coords, steps = load_data(mdname, atom_names, np.sum(atom_numbs))
    data, steps = to_system_data(fname, mdname)
    energy, force = to_system_label(fname, mdname)
    print(coords)
    print(energy)
    # print(atom_names)
    # print(atom_numbs)
    # print(atom_types)
    # print(cells.shape)
    # print(coords.shape)
    # print(len(energy))
    # print(force.shape)
