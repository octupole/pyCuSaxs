#!/usr/bin/env python3
"""
Tool to subtract solvent SAXS profile from protein+solvent profile.

This computes: I_protein(q) = I_total(q) - I_solvent(q)

Usage:
    python -m pycusaxs.saxs_subtract --protein-id 10 --solvent-id 2 --output protein_only.dat
    python -m pycusaxs.saxs_subtract --protein-file protein_solvent.dat --solvent-file water.dat --output protein_only.dat
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from .saxs_database import SaxsDatabase
from .saxs_defaults import SaxsDefaults


def read_saxs_dat(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read SAXS profile from .dat file.

    Args:
        file_path: Path to .dat file

    Returns:
        Tuple of (q_values, intensity_values)
    """
    q_values = []
    i_values = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    q = float(parts[0])
                    iq = float(parts[1])
                    q_values.append(q)
                    i_values.append(iq)
                except ValueError:
                    continue

    return np.array(q_values), np.array(i_values)


def interpolate_profile(q_target: np.ndarray, q_source: np.ndarray,
                       i_source: np.ndarray) -> np.ndarray:
    """
    Interpolate intensity values to match target q-values.

    Args:
        q_target: Target q-values
        q_source: Source q-values
        i_source: Source intensity values

    Returns:
        Interpolated intensity values at q_target
    """
    return np.interp(q_target, q_source, i_source, left=0.0, right=0.0)


def subtract_profiles(q_protein: np.ndarray, i_protein: np.ndarray,
                     q_solvent: np.ndarray, i_solvent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subtract solvent profile from protein+solvent profile.

    Args:
        q_protein: Q-values from protein+solvent
        i_protein: Intensity from protein+solvent
        q_solvent: Q-values from pure solvent
        i_solvent: Intensity from pure solvent

    Returns:
        Tuple of (q_values, subtracted_intensity)
    """
    # Interpolate solvent profile to match protein q-grid
    i_solvent_interp = interpolate_profile(q_protein, q_solvent, i_solvent)

    # Subtract
    i_diff = i_protein - i_solvent_interp

    return q_protein, i_diff


def write_saxs_profile(file_path: str, q_values: np.ndarray, i_values: np.ndarray,
                      header: Optional[List[str]] = None):
    """
    Write SAXS profile to file.

    Args:
        file_path: Output file path
        q_values: Q-values
        i_values: Intensity values
        header: Optional header lines
    """
    with open(file_path, 'w') as f:
        if header:
            for line in header:
                f.write(f"# {line}\n")

        f.write("# q (1/Å), I(q) (1/Å³)\n")

        for q, iq in zip(q_values, i_values):
            f.write(f"{q:.6f}\t{iq:.6e}\n")


def subtract_from_database(protein_id: int, solvent_id: int, output_file: str,
                          user_db: str, reference_db: str):
    """
    Subtract profiles from databases.

    Args:
        protein_id: Profile ID in user database
        solvent_id: Profile ID in reference database
        output_file: Output file path
        user_db: Path to user database
        reference_db: Path to reference database
    """
    # Load protein+solvent profile from user database
    with SaxsDatabase(user_db) as db:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM saxs_profiles WHERE id = ?", (protein_id,))
        row = cursor.fetchone()

        if not row:
            raise ValueError(f"Protein profile ID {protein_id} not found in user database")

        protein_profile = db._row_to_dict(row)

    # Load solvent profile from reference database
    with SaxsDatabase(reference_db) as db:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM saxs_profiles WHERE id = ?", (solvent_id,))
        row = cursor.fetchone()

        if not row:
            raise ValueError(f"Solvent profile ID {solvent_id} not found in reference database")

        solvent_profile = db._row_to_dict(row)

    # Extract data
    protein_data = np.array(protein_profile['profile_data'])
    solvent_data = np.array(solvent_profile['profile_data'])

    q_protein = protein_data[:, 0]
    i_protein = protein_data[:, 1]
    q_solvent = solvent_data[:, 0]
    i_solvent = solvent_data[:, 1]

    # Subtract
    q_result, i_result = subtract_profiles(q_protein, i_protein, q_solvent, i_solvent)

    # Generate header
    header = [
        "SAXS Profile Subtraction",
        f"Protein+Solvent Profile ID: {protein_id} (from user database)",
        f"  Water Model: {protein_profile['water_model']}",
        f"  Box: {protein_profile['box_x']:.1f} x {protein_profile['box_y']:.1f} x {protein_profile['box_z']:.1f} Å",
        f"  Supercell Scale: {protein_profile['supercell_scale']:.4f}",
        "",
        f"Pure Solvent Profile ID: {solvent_id} (from reference database)",
        f"  Water Model: {solvent_profile['water_model']}",
        f"  Box: {solvent_profile['box_x']:.1f} x {solvent_profile['box_y']:.1f} x {solvent_profile['box_z']:.1f} Å",
        f"  Supercell Scale: {solvent_profile['supercell_scale']:.4f}",
        "",
        "Result: I_protein(q) = I_total(q) - I_solvent(q)",
    ]

    # Write output
    write_saxs_profile(output_file, q_result, i_result, header)

    print(f"\nSubtraction completed successfully!")
    print(f"Input: Protein+Solvent (ID {protein_id}) - Solvent (ID {solvent_id})")
    print(f"Output: {output_file}")
    print(f"Data points: {len(q_result)}")


def subtract_from_files(protein_file: str, solvent_file: str, output_file: str):
    """
    Subtract profiles from .dat files.

    Args:
        protein_file: Path to protein+solvent .dat file
        solvent_file: Path to pure solvent .dat file
        output_file: Output file path
    """
    # Read files
    q_protein, i_protein = read_saxs_dat(protein_file)
    q_solvent, i_solvent = read_saxs_dat(solvent_file)

    # Subtract
    q_result, i_result = subtract_profiles(q_protein, i_protein, q_solvent, i_solvent)

    # Generate header
    header = [
        "SAXS Profile Subtraction",
        f"Protein+Solvent File: {protein_file}",
        f"Pure Solvent File: {solvent_file}",
        "",
        "Result: I_protein(q) = I_total(q) - I_solvent(q)",
    ]

    # Write output
    write_saxs_profile(output_file, q_result, i_result, header)

    print(f"\nSubtraction completed successfully!")
    print(f"Input: {protein_file} - {solvent_file}")
    print(f"Output: {output_file}")
    print(f"Data points: {len(q_result)}")


def main():
    parser = argparse.ArgumentParser(
        description="Subtract solvent SAXS profile from protein+solvent profile"
    )

    # Database mode
    db_group = parser.add_argument_group("Database mode (subtract profiles from databases)")
    db_group.add_argument(
        "--protein-id",
        type=int,
        help="Profile ID in user database (protein+solvent system)"
    )
    db_group.add_argument(
        "--solvent-id",
        type=int,
        help="Profile ID in reference database (pure solvent system)"
    )
    db_group.add_argument(
        "--user-db",
        default=str(SaxsDefaults.get_user_database_path()),
        help=f"User database path (default: {SaxsDefaults.get_user_database_path()})"
    )
    db_group.add_argument(
        "--reference-db",
        default=str(SaxsDefaults.get_reference_database_path()),
        help=f"Reference database path (default: {SaxsDefaults.get_reference_database_path()})"
    )

    # File mode
    file_group = parser.add_argument_group("File mode (subtract .dat files)")
    file_group.add_argument(
        "--protein-file",
        help="Path to protein+solvent .dat file"
    )
    file_group.add_argument(
        "--solvent-file",
        help="Path to pure solvent .dat file"
    )

    # Common arguments
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path for subtracted profile"
    )

    args = parser.parse_args()

    # Validate arguments
    db_mode = args.protein_id is not None and args.solvent_id is not None
    file_mode = args.protein_file is not None and args.solvent_file is not None

    if not db_mode and not file_mode:
        parser.error("Must specify either --protein-id/--solvent-id OR --protein-file/--solvent-file")

    if db_mode and file_mode:
        parser.error("Cannot use both database mode and file mode simultaneously")

    try:
        if db_mode:
            subtract_from_database(
                args.protein_id,
                args.solvent_id,
                args.output,
                args.user_db,
                args.reference_db
            )
        else:
            subtract_from_files(
                args.protein_file,
                args.solvent_file,
                args.output
            )

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
