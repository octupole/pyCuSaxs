#!/usr/bin/env python3
"""
Interactive tool for subtracting reference solvent SAXS profiles from simulated profiles.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
from scipy.interpolate import interp1d

from pycusaxs.saxs_database import SaxsDatabase
from pycusaxs.saxs_defaults import SaxsDefaults


def _format_density(value) -> str:
    """Return density as formatted string handling missing values."""
    return f"{value:.4f}" if value is not None else "N/A"


def list_profiles(db: SaxsDatabase, title: str) -> List[dict]:
    """List profiles in a database and return them."""
    profiles = db.list_profiles()

    if not profiles:
        print(f"\n{title}: No profiles found.")
        return []

    print(f"\n{title}:")
    print(f"{'ID':<5} {'Water':<8} {'Ions':<20} {'Other Molecules':<25} {'Grid':<13} {'Supercell':<13} {'Time (ps)':<11} {'Density':<10}")
    print("-" * 130)

    for profile in profiles:
        grid = profile['grid_size']
        grid_str = f"{grid[0]}x{grid[1]}x{grid[2]}"

        scale = profile['supercell_scale']
        supercell_str = f"{int(grid[0]*scale)}x{int(grid[1]*scale)}x{int(grid[2]*scale)}"

        # Format ions
        ion_counts = profile.get('ion_counts', {})
        if ion_counts:
            ion_str = ', '.join([f"{k}:{v}" for k, v in ion_counts.items() if v > 0])
            if not ion_str:
                ion_str = "none"
        else:
            ion_str = "none"

        # Format other molecules
        other_mols = profile.get('other_molecules', {})
        if other_mols:
            mol_str = ', '.join([f"{k}:{v}" for k, v in other_mols.items()])[:24]  # Truncate if too long
        else:
            mol_str = "none"

        density_str = _format_density(profile.get('density_g_cm3'))

        print(f"{profile['id']:<5} {profile['water_model']:<8} {ion_str:<20} {mol_str:<25} "
              f"{grid_str:<13} {supercell_str:<13} {profile['simulation_time_ps']:<11.2f} {density_str:<10}")

    return profiles


def interpolate_profile(q_ref: np.ndarray, iq_ref: np.ndarray,
                        q_target: np.ndarray, method: str = 'cubic') -> np.ndarray:
    """
    Interpolate reference profile to match target q-grid.

    Args:
        q_ref: Reference q values
        iq_ref: Reference I(q) values
        q_target: Target q values
        method: Interpolation method ('linear' or 'cubic')

    Returns:
        Interpolated I(q) values on target grid
    """
    if method == 'cubic':
        interpolator = interp1d(q_ref, iq_ref, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')
    else:
        interpolator = interp1d(q_ref, iq_ref, kind='linear',
                                bounds_error=False, fill_value='extrapolate')

    return interpolator(q_target)


def get_user_choice(prompt: str, min_val: int, max_val: int) -> int:
    """Get user input within a valid range."""
    while True:
        try:
            choice = int(input(prompt))
            if min_val <= choice <= max_val:
                return choice
            print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(0)


def get_scaling_factor() -> float:
    """
    Get the scaling factor for subtraction from user input.

    Returns:
        Scaling factor
    """
    while True:
        try:
            scale_str = input(
                "\nEnter scaling factor for reference subtraction: ").strip()
            scale = float(scale_str)
            if scale > 0:
                return scale
            print("Scaling factor must be positive.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(0)


def subtract_profiles(user_profile: dict, ref_profile: dict,
                      scale: float, interp_method: str = 'cubic') -> Tuple[np.ndarray, np.ndarray]:
    """
    Subtract scaled reference profile from user profile.

    Args:
        user_profile: User's simulated profile
        ref_profile: Reference solvent profile
        scale: Scaling factor for reference
        interp_method: Interpolation method if grids differ

    Returns:
        (q, I_subtracted) arrays
    """
    # Extract profiles
    user_data = np.array(user_profile['profile_data'])
    ref_data = np.array(ref_profile['profile_data'])

    q_user = user_data[:, 0]
    iq_user = user_data[:, 1]

    q_ref = ref_data[:, 0]
    iq_ref = ref_data[:, 1]

    # Check if grids match
    grids_match = (len(q_user) == len(q_ref) and
                   np.allclose(q_user, q_ref, rtol=1e-6))

    if grids_match:
        print(f"\nQ-grids match exactly ({len(q_user)} points)")
        iq_ref_interp = iq_ref
    else:
        print(
            f"\nQ-grids differ: user has {len(q_user)} points, reference has {len(q_ref)} points")
        print(f"User q range: [{q_user[0]:.6f}, {q_user[-1]:.6f}] Å⁻¹")
        print(f"Reference q range: [{q_ref[0]:.6f}, {q_ref[-1]:.6f}] Å⁻¹")
        print(
            f"Interpolating reference profile using {interp_method} interpolation...")

        iq_ref_interp = interpolate_profile(
            q_ref, iq_ref, q_user, method=interp_method)

    # Perform subtraction
    iq_subtracted = iq_user - scale * iq_ref_interp

    return q_user, iq_subtracted


def resample_profile(q: np.ndarray, iq: np.ndarray, dq: float,
                     method: str = 'cubic') -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample profile to a uniform q-grid with specified spacing.

    Args:
        q: Original q values
        iq: Original I(q) values
        dq: Desired q spacing (Å⁻¹)
        method: Interpolation method ('linear' or 'cubic')

    Returns:
        (q_resampled, iq_resampled) arrays
    """
    # Create uniform q-grid from min to max with spacing dq
    q_min = q[0]
    q_max = q[-1]
    n_points = int(np.ceil((q_max - q_min) / dq)) + 1
    q_uniform = np.linspace(q_min, q_max, n_points)

    print(f"\nResampling profile to uniform grid:")
    print(f"  Original points: {len(q)}")
    print(f"  New dq: {dq:.6f} Å⁻¹")
    print(f"  New points: {len(q_uniform)}")
    print(f"  Q range: [{q_uniform[0]:.6f}, {q_uniform[-1]:.6f}] Å⁻¹")

    # Interpolate I(q) to the uniform grid
    iq_uniform = interpolate_profile(q, iq, q_uniform, method=method)

    return q_uniform, iq_uniform


def save_subtracted_profile(output_path: Path, q: np.ndarray, iq: np.ndarray,
                            user_profile: dict, ref_profile: dict, scale: float,
                            resampled: bool = False, original_points: int = None):
    """
    Save subtracted profile to file with metadata.

    Args:
        output_path: Path to output file
        q: q values
        iq: I(q) values
        user_profile: User profile dictionary
        ref_profile: Reference profile dictionary
        scale: Scaling factor used
        resampled: Whether the profile was resampled
        original_points: Number of points before resampling (if resampled)
    """
    with open(output_path, 'w') as f:
        f.write("# SAXS Profile after Solvent Subtraction\n")
        f.write("#\n")
        f.write("# User Profile:\n")
        f.write(f"#   ID: {user_profile['id']}\n")
        f.write(f"#   Water Model: {user_profile['water_model']}\n")

        # Write ions for user profile
        user_ions = user_profile.get('ion_counts', {})
        if user_ions:
            ions_str = ', '.join([f"{k}: {v}" for k, v in user_ions.items() if v > 0])
            f.write(f"#   Ions: {ions_str if ions_str else 'none'}\n")
        else:
            f.write(f"#   Ions: none\n")

        # Write other molecules for user profile
        user_other = user_profile.get('other_molecules', {})
        if user_other:
            mols_str = ', '.join([f"{k}: {v}" for k, v in user_other.items()])
            f.write(f"#   Other Molecules: {mols_str}\n")
        else:
            f.write(f"#   Other Molecules: none\n")

        f.write(f"#   Grid: {user_profile['grid_size']}\n")
        f.write(
            f"#   Supercell Scale: {user_profile['supercell_scale']:.4f}\n")
        f.write(
            f"#   Box: {user_profile['box_x']:.3f} x {user_profile['box_y']:.3f} x {user_profile['box_z']:.3f} Å\n")
        f.write(f"#   Volume: {user_profile['box_volume']:.2f} Ų\n")
        user_density = _format_density(user_profile.get('density_g_cm3'))
        f.write(f"#   Density: {user_density} g/cm³\n")
        f.write(
            f"#   Simulation Time: {user_profile['simulation_time_ps']:.2f} ps\n")
        f.write("#\n")
        f.write("# Reference Profile (subtracted):\n")
        f.write(f"#   ID: {ref_profile['id']}\n")
        f.write(f"#   Water Model: {ref_profile['water_model']}\n")

        # Write ions for reference profile
        ref_ions = ref_profile.get('ion_counts', {})
        if ref_ions:
            ions_str = ', '.join([f"{k}: {v}" for k, v in ref_ions.items() if v > 0])
            f.write(f"#   Ions: {ions_str if ions_str else 'none'}\n")
        else:
            f.write(f"#   Ions: none\n")

        # Write other molecules for reference profile
        ref_other = ref_profile.get('other_molecules', {})
        if ref_other:
            mols_str = ', '.join([f"{k}: {v}" for k, v in ref_other.items()])
            f.write(f"#   Other Molecules: {mols_str}\n")
        else:
            f.write(f"#   Other Molecules: none\n")

        f.write(f"#   Grid: {ref_profile['grid_size']}\n")
        f.write(f"#   Supercell Scale: {ref_profile['supercell_scale']:.4f}\n")
        f.write(
            f"#   Box: {ref_profile['box_x']:.3f} x {ref_profile['box_y']:.3f} x {ref_profile['box_z']:.3f} Å\n")
        f.write(f"#   Volume: {ref_profile['box_volume']:.2f} Ų\n")
        ref_density = _format_density(ref_profile.get('density_g_cm3'))
        f.write(f"#   Density: {ref_density} g/cm³\n")
        f.write(
            f"#   Simulation Time: {ref_profile['simulation_time_ps']:.2f} ps\n")
        f.write(f"#   Scaling Factor: {scale:.6f}\n")
        f.write("#\n")
        if resampled and original_points is not None:
            f.write("# Output Processing:\n")
            f.write(f"#   Original data points: {original_points}\n")
            f.write(f"#   Resampled to: {len(q)} points\n")
            f.write(f"#   Q spacing (dq): {(q[1] - q[0]):.6f} Å⁻¹\n")
            f.write("#\n")
        f.write("# q (Å⁻¹)    I(q) [subtracted]\n")

        for q_val, iq_val in zip(q, iq):
            f.write(f"{q_val:.6f}  {iq_val:.6e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Subtract reference solvent SAXS profile from simulated profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (choose profile ID interactively)
  saxs-subtract --db my_data.db

  # Specify profile ID directly
  saxs-subtract --db my_data.db --id 1

  # Specify reference database
  saxs-subtract --db my_data.db --ref-db reference.db

  # Specify output file
  saxs-subtract --db my_data.db --id 1 -o subtracted.dat

  # Resample to uniform q-grid with dq=0.01 Å⁻¹
  saxs-subtract --db my_data.db --id 1 --dq 0.01

  # Use linear interpolation instead of cubic
  saxs-subtract --db my_data.db --id 1 --interp linear
        """
    )

    parser.add_argument("--db", required=True, type=str,
                        help="Path to user database containing simulated profile")
    parser.add_argument("--id", type=int,
                        help="Profile ID from user database to process (interactive if not specified)")
    parser.add_argument("--ref-db", type=str,
                        help="Path to reference database (default: package reference database)")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file path (default: subtracted_<id>.dat)")
    parser.add_argument("--dq", type=float,
                        help="Resample output to uniform q-grid with this spacing (Å⁻¹)")
    parser.add_argument("--interp", choices=['linear', 'cubic'], default='cubic',
                        help="Interpolation method if q-grids differ (default: cubic)")

    args = parser.parse_args()

    # Determine reference database path
    if args.ref_db:
        ref_db_path = Path(args.ref_db).expanduser().resolve()
    else:
        ref_db_path = SaxsDefaults.get_reference_database_path()

    if not ref_db_path.exists():
        print(
            f"Error: Reference database not found: {ref_db_path}", file=sys.stderr)
        return 1

    user_db_path = Path(args.db).expanduser().resolve()
    if not user_db_path.exists():
        print(
            f"Error: User database not found: {user_db_path}", file=sys.stderr)
        return 1

    # Load and display simulated profiles from user database
    print(f"\nLoading simulated profiles from: {user_db_path}")
    with SaxsDatabase(user_db_path) as user_db:
        user_profiles = list_profiles(user_db, "Available Simulated Profiles")

        if not user_profiles:
            print("\nError: No simulated profiles available.", file=sys.stderr)
            return 1

        # Get profile ID (from argument or interactively)
        if args.id is not None:
            user_id = args.id
            # Verify the ID exists
            user_profile = user_db.get_profile(user_id)
            if not user_profile:
                print(
                    f"Error: Profile ID {user_id} not found in user database.", file=sys.stderr)
                return 1
        else:
            # Interactive selection
            user_id = get_user_choice(
                f"\nSelect simulated profile ID to subtract from: ",
                min(p['id'] for p in user_profiles),
                max(p['id'] for p in user_profiles)
            )
            user_profile = user_db.get_profile(user_id)
            if not user_profile:
                print(
                    f"Error: Could not load profile {user_id}.", file=sys.stderr)
                return 1

        print(f"\nSelected Simulated Profile {user_id}:")
        print(f"  Water Model: {user_profile['water_model']}")

        # Show ions
        ion_counts = user_profile.get('ion_counts', {})
        if ion_counts:
            ions_list = [f"{k}: {v}" for k, v in ion_counts.items() if v > 0]
            if ions_list:
                print(f"  Ions: {', '.join(ions_list)}")
            else:
                print(f"  Ions: none")
        else:
            print(f"  Ions: none")

        # Show other molecules
        other_mols = user_profile.get('other_molecules', {})
        if other_mols:
            mols_list = [f"{k}: {v}" for k, v in other_mols.items()]
            print(f"  Other Molecules: {', '.join(mols_list)}")
        else:
            print(f"  Other Molecules: none")

        print(f"  Grid: {user_profile['grid_size']}")
        print(f"  Supercell Scale: {user_profile['supercell_scale']:.4f}")
        print(
            f"  Box: {user_profile['box_x']:.3f} x {user_profile['box_y']:.3f} x {user_profile['box_z']:.3f} Å")
        print(f"  Volume: {user_profile['box_volume']:.2f} Ų")
        density_str = _format_density(user_profile.get('density_g_cm3'))
        print(f"  Density: {density_str} g/cm³")
        print(
            f"  Simulation Time: {user_profile['simulation_time_ps']:.2f} ps")

    # List reference profiles and get user selection
    print(f"\nReference database: {ref_db_path}")
    with SaxsDatabase(ref_db_path) as ref_db:
        ref_profiles = list_profiles(ref_db, "Available Reference Profiles")

        if not ref_profiles:
            print("\nError: No reference profiles available.", file=sys.stderr)
            return 1

        # User selects reference profile
        ref_id = get_user_choice(
            f"\nSelect reference profile ID (1-{len(ref_profiles)}): ",
            1, len(ref_profiles)
        )

        ref_profile = ref_db.get_profile(ref_id)
        if not ref_profile:
            print(
                f"Error: Could not load reference profile {ref_id}.", file=sys.stderr)
            return 1

    # Get scaling factor
    scale = get_scaling_factor()

    print(
        f"\nSubtracting reference profile {ref_id} (scaled by {scale:.6f})...")

    # Perform subtraction
    q, iq_subtracted = subtract_profiles(user_profile, ref_profile, scale,
                                         interp_method=args.interp)

    # Resample if --dq is specified
    resampled = False
    original_points = None
    if args.dq is not None:
        if args.dq <= 0:
            print(f"Error: --dq must be positive (got {args.dq})", file=sys.stderr)
            return 1

        original_points = len(q)
        q, iq_subtracted = resample_profile(q, iq_subtracted, args.dq,
                                           method=args.interp)
        resampled = True

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"subtracted_{user_id}.dat")

    # Save result
    save_subtracted_profile(output_path, q, iq_subtracted,
                            user_profile, ref_profile, scale,
                            resampled=resampled, original_points=original_points)

    print(f"\nSubtracted profile saved to: {output_path}")
    print(f"Data points: {len(q)}")
    print(f"Q range: [{q[0]:.6f}, {q[-1]:.6f}] Å⁻¹")

    # Show some statistics
    print(f"\nIntensity statistics after subtraction:")
    print(f"  Min I(q): {np.min(iq_subtracted):.6e}")
    print(f"  Max I(q): {np.max(iq_subtracted):.6e}")
    print(f"  Mean I(q): {np.mean(iq_subtracted):.6e}")

    # Warn if many negative values
    n_negative = np.sum(iq_subtracted < 0)
    if n_negative > 0:
        pct_negative = 100 * n_negative / len(iq_subtracted)
        print(
            f"\nWarning: {n_negative} ({pct_negative:.1f}%) data points are negative.")
        print(f"This may indicate over-subtraction (scaling factor too large).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
