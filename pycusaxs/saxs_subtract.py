#!/usr/bin/env python3
"""
Interactive tool for subtracting reference solvent SAXS profiles from experimental profiles.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
from scipy.interpolate import interp1d

from pycusaxs.saxs_database import SaxsDatabase
from pycusaxs.saxs_defaults import SaxsDefaults


def list_profiles(db: SaxsDatabase, title: str) -> List[dict]:
    """List profiles in a database and return them."""
    profiles = db.list_profiles()

    if not profiles:
        print(f"\n{title}: No profiles found.")
        return []

    print(f"\n{title}:")
    print(f"{'ID':<5} {'Water Model':<15} {'Grid':<15} {'Supercell':<15} {'Time (ps)':<12} {'Density (g/cm³)':<18}")
    print("-" * 100)

    for profile in profiles:
        grid = profile['grid_size']
        grid_str = f"{grid[0]}x{grid[1]}x{grid[2]}"

        scale = profile['supercell_scale']
        supercell_str = f"{int(grid[0]*scale)}x{int(grid[1]*scale)}x{int(grid[2]*scale)}"

        print(f"{profile['id']:<5} {profile['water_model']:<15} {grid_str:<15} {supercell_str:<15} "
              f"{profile['simulation_time_ps']:<12.2f} {profile['density_g_cm3']:<18.4f}")

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


def get_scaling_factor(user_profile: dict, ref_profile: dict, auto: bool = False) -> float:
    """
    Get or calculate the scaling factor for subtraction.

    Args:
        user_profile: User's experimental profile
        ref_profile: Reference solvent profile
        auto: If True, auto-calculate from densities and volumes

    Returns:
        Scaling factor
    """
    if auto:
        # Estimate scaling from relative volumes and densities
        user_vol = user_profile['box_volume']
        ref_vol = ref_profile['box_volume']
        user_dens = user_profile['density_g_cm3']
        ref_dens = ref_profile['density_g_cm3']

        # Scale by volume ratio (assuming same number of water molecules should scale with volume)
        scale = user_vol / ref_vol if ref_vol > 0 else 1.0

        print(f"\nAuto-calculated scaling factor: {scale:.6f}")
        print(f"  Based on volume ratio: {user_vol:.2f} / {ref_vol:.2f} Ų")

        response = input("Use this scaling factor? [Y/n]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            return scale

    # Manual input
    while True:
        try:
            scale_str = input("\nEnter scaling factor for reference subtraction: ").strip()
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
        user_profile: User's experimental profile
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
        print(f"\nQ-grids differ: user has {len(q_user)} points, reference has {len(q_ref)} points")
        print(f"User q range: [{q_user[0]:.6f}, {q_user[-1]:.6f}] Å⁻¹")
        print(f"Reference q range: [{q_ref[0]:.6f}, {q_ref[-1]:.6f}] Å⁻¹")
        print(f"Interpolating reference profile using {interp_method} interpolation...")

        iq_ref_interp = interpolate_profile(q_ref, iq_ref, q_user, method=interp_method)

    # Perform subtraction
    iq_subtracted = iq_user - scale * iq_ref_interp

    return q_user, iq_subtracted


def save_subtracted_profile(output_path: Path, q: np.ndarray, iq: np.ndarray,
                           user_profile: dict, ref_profile: dict, scale: float):
    """Save subtracted profile to file with metadata."""
    with open(output_path, 'w') as f:
        f.write("# SAXS Profile after Solvent Subtraction\n")
        f.write("#\n")
        f.write("# User Profile:\n")
        f.write(f"#   ID: {user_profile['id']}\n")
        f.write(f"#   Water Model: {user_profile['water_model']}\n")
        f.write(f"#   Grid: {user_profile['grid_size']}\n")
        f.write(f"#   Supercell Scale: {user_profile['supercell_scale']:.4f}\n")
        f.write(f"#   Box: {user_profile['box_x']:.3f} x {user_profile['box_y']:.3f} x {user_profile['box_z']:.3f} Å\n")
        f.write(f"#   Volume: {user_profile['box_volume']:.2f} Ų\n")
        f.write(f"#   Density: {user_profile['density_g_cm3']:.4f} g/cm³\n")
        f.write(f"#   Simulation Time: {user_profile['simulation_time_ps']:.2f} ps\n")
        f.write("#\n")
        f.write("# Reference Profile (subtracted):\n")
        f.write(f"#   ID: {ref_profile['id']}\n")
        f.write(f"#   Water Model: {ref_profile['water_model']}\n")
        f.write(f"#   Grid: {ref_profile['grid_size']}\n")
        f.write(f"#   Supercell Scale: {ref_profile['supercell_scale']:.4f}\n")
        f.write(f"#   Box: {ref_profile['box_x']:.3f} x {ref_profile['box_y']:.3f} x {ref_profile['box_z']:.3f} Å\n")
        f.write(f"#   Volume: {ref_profile['box_volume']:.2f} Ų\n")
        f.write(f"#   Density: {ref_profile['density_g_cm3']:.4f} g/cm³\n")
        f.write(f"#   Simulation Time: {ref_profile['simulation_time_ps']:.2f} ps\n")
        f.write(f"#   Scaling Factor: {scale:.6f}\n")
        f.write("#\n")
        f.write("# q (Å⁻¹)    I(q) [subtracted]\n")

        for q_val, iq_val in zip(q, iq):
            f.write(f"{q_val:.6f}  {iq_val:.6e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Subtract reference solvent SAXS profile from experimental profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  saxs-subtract --db my_data.db --id 1

  # Specify reference database and auto-calculate scaling
  saxs-subtract --db my_data.db --id 1 --ref-db reference.db --auto-scale

  # Specify output file
  saxs-subtract --db my_data.db --id 1 -o subtracted.dat

  # Use linear interpolation instead of cubic
  saxs-subtract --db my_data.db --id 1 --interp linear
        """
    )

    parser.add_argument("--db", required=True, type=str,
                       help="Path to user database containing experimental profile")
    parser.add_argument("--id", required=True, type=int,
                       help="Profile ID from user database to process")
    parser.add_argument("--ref-db", type=str,
                       help="Path to reference database (default: package reference database)")
    parser.add_argument("-o", "--output", type=str,
                       help="Output file path (default: subtracted_<id>.dat)")
    parser.add_argument("--auto-scale", action="store_true",
                       help="Auto-calculate scaling factor from volumes and densities")
    parser.add_argument("--interp", choices=['linear', 'cubic'], default='cubic',
                       help="Interpolation method if q-grids differ (default: cubic)")

    args = parser.parse_args()

    # Determine reference database path
    if args.ref_db:
        ref_db_path = Path(args.ref_db).expanduser().resolve()
    else:
        ref_db_path = SaxsDefaults.get_reference_database_path()

    if not ref_db_path.exists():
        print(f"Error: Reference database not found: {ref_db_path}", file=sys.stderr)
        return 1

    user_db_path = Path(args.db).expanduser().resolve()
    if not user_db_path.exists():
        print(f"Error: User database not found: {user_db_path}", file=sys.stderr)
        return 1

    # Load user profile
    print(f"\nLoading user profile from: {user_db_path}")
    with SaxsDatabase(user_db_path) as user_db:
        user_profile = user_db.get_profile(args.id)
        if not user_profile:
            print(f"Error: Profile ID {args.id} not found in user database.", file=sys.stderr)
            return 1

        print(f"\nUser Profile {args.id}:")
        print(f"  Water Model: {user_profile['water_model']}")
        print(f"  Grid: {user_profile['grid_size']}")
        print(f"  Supercell Scale: {user_profile['supercell_scale']:.4f}")
        print(f"  Box: {user_profile['box_x']:.3f} x {user_profile['box_y']:.3f} x {user_profile['box_z']:.3f} Å")
        print(f"  Volume: {user_profile['box_volume']:.2f} Ų")
        print(f"  Density: {user_profile['density_g_cm3']:.4f} g/cm³")
        print(f"  Simulation Time: {user_profile['simulation_time_ps']:.2f} ps")

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
            print(f"Error: Could not load reference profile {ref_id}.", file=sys.stderr)
            return 1

    # Get scaling factor
    scale = get_scaling_factor(user_profile, ref_profile, auto=args.auto_scale)

    print(f"\nSubtracting reference profile {ref_id} (scaled by {scale:.6f})...")

    # Perform subtraction
    q, iq_subtracted = subtract_profiles(user_profile, ref_profile, scale,
                                        interp_method=args.interp)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"subtracted_{args.id}.dat")

    # Save result
    save_subtracted_profile(output_path, q, iq_subtracted,
                           user_profile, ref_profile, scale)

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
        print(f"\nWarning: {n_negative} ({pct_negative:.1f}%) data points are negative.")
        print(f"This may indicate over-subtraction (scaling factor too large).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
