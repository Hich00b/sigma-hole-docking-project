"""Consolidated PDBQT parsing and writing utilities.

All modules that need to read or write PDBQT files should use these functions
instead of maintaining their own parsers. This eliminates the divergent bugs
that existed when each module parsed PDBQT independently (e.g. reading the
temperature-factor column as charge).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# PDBQT column layout (whitespace-split):
#   0: record type (ATOM / HETATM)
#   1: serial
#   2: atom name
#   3: residue name
#   4: chain id
#   5: residue sequence
#   6-8: x, y, z
#   9: occupancy
#  10: temperature factor
#  11: partial charge
#  12: atom type (optional)

_ELEMENT_MAP = {
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "AT": "At",
}


def normalize_element(element: str) -> str:
    """Normalize an element symbol to standard capitalization (e.g. ``CL`` → ``Cl``)."""
    return _ELEMENT_MAP.get(element.upper(), element)


def is_dummy_atom(atom_type: str, element: str) -> bool:
    """Return True if the atom is a sigma-hole dummy (extra-point) charge site."""
    return atom_type == "EP"


def parse_pdbqt(pdbqt_path: str) -> List[Dict]:
    """Parse a PDBQT file and return a list of atom dictionaries.

    Each dictionary has keys:
        index, element, x, y, z, charge, atom_type, is_dummy

    Malformed lines are skipped with a debug log.  A missing file or IO error
    returns an empty list (callers should check ``len(atoms) == 0``).
    """
    atoms: List[Dict] = []
    parsing_errors = 0

    try:
        with open(pdbqt_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue

                parts = line.split()
                if len(parts) < 12:
                    parsing_errors += 1
                    if parsing_errors <= 5:
                        logger.debug(
                            "Malformed ATOM/HETATM line %d (too few fields): %s",
                            line_num,
                            line,
                        )
                    continue

                try:
                    atom_type = parts[12] if len(parts) > 12 else parts[2]
                    element = normalize_element(parts[2])
                    atoms.append(
                        {
                            "index": int(parts[1]),
                            "element": element,
                            "x": float(parts[6]),
                            "y": float(parts[7]),
                            "z": float(parts[8]),
                            "charge": float(parts[11]),
                            "atom_type": atom_type,
                            "is_dummy": is_dummy_atom(atom_type, element),
                        }
                    )
                except (ValueError, IndexError) as exc:
                    parsing_errors += 1
                    if parsing_errors <= 5:
                        logger.debug(
                            "Could not parse ATOM/HETATM line %d: %s (%s)",
                            line_num,
                            line,
                            exc,
                        )
                    continue

        if parsing_errors > 5:
            logger.debug("... and %d more PDBQT parsing errors", parsing_errors - 5)

    except FileNotFoundError:
        logger.error("PDBQT file not found: %s", pdbqt_path)
        return []
    except Exception as exc:
        logger.error("Error reading PDBQT file %s: %s", pdbqt_path, exc)
        return []

    if not atoms:
        logger.warning("No ATOM or HETATM records found in PDBQT file: %s", pdbqt_path)

    logger.debug("Parsed %d atoms from %s", len(atoms), pdbqt_path)
    return atoms


def write_pdbqt(
    atoms: List[Dict],
    output_path: str,
    remarks: Optional[List[str]] = None,
    residue_name: str = "LIG",
    chain_id: str = "B",
    residue_seq: int = 1,
    torsdof: int = 0,
) -> None:
    """Write a list of atom dictionaries to a PDBQT file.

    Each atom dict should have keys: ``element``, ``x``, ``y``, ``z``,
    ``charge``, ``atom_type``.  ``index`` is auto-numbered if absent.

    Args:
        atoms: List of atom dictionaries.
        output_path: Destination file path.
        remarks: Optional list of REMARK lines (without the ``REMARK`` prefix).
        residue_name: 3-letter residue name (default ``LIG``).
        chain_id: Single-character chain ID (default ``B``).
        residue_seq: Residue sequence number (default 1).
        torsdof: Torsional degrees of freedom (default 0).
    """
    with open(output_path, "w") as f:
        if remarks:
            for remark in remarks:
                f.write(f"REMARK {remark}\n")
        f.write("ROOT\n")

        for i, atom in enumerate(atoms):
            element = atom.get("element", "C")
            atom_type = atom.get("atom_type", element)
            charge = atom.get("charge", 0.0)
            idx = atom.get("index", i + 1)
            f.write(
                f"ATOM {idx:4d} {element:<2s} {residue_name} {chain_id} {residue_seq:3d} "
                f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f} "
                f"0.00 0.00 {charge:7.4f} {atom_type:2s}\n"
            )

        f.write("ENDROOT\n")
        f.write("TORSDOF\n")
        f.write(f"{torsdof}\n")

    logger.debug("Wrote %d atoms to %s", len(atoms), output_path)
