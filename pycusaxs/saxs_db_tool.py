#!/usr/bin/env python3
"""
Command-line tool for managing SAXS profile database.

Usage:
    python -m pycusaxs.saxs_db_tool list [--db DATABASE]
    python -m pycusaxs.saxs_db_tool export ID OUTPUT_CSV [--db DATABASE]
    python -m pycusaxs.saxs_db_tool info ID [--db DATABASE]
"""

import argparse
import sys
from .saxs_database import SaxsDatabase
from .saxs_defaults import SaxsDefaults


def cmd_list(args):
    """List all profiles in database."""
    with SaxsDatabase(args.db) as db:
        profiles = db.list_profiles(water_model=args.water_model)

        if not profiles:
            print("No profiles found in database.")
            return

        print(f"\n{'='*80}")
        print(f"SAXS Profiles in {args.db}")
        print(f"{'='*80}")
        print(f"{'ID':<5} {'Water':<8} {'Ions':<20} {'Box (Å)':<25} {'Scale':<8} {'Time (ps)':<10}")
        print(f"{'-'*80}")

        for profile in profiles:
            ion_str = ', '.join([f"{k}:{v}" for k, v in profile['ion_counts'].items() if v > 0])
            if not ion_str:
                ion_str = "none"

            box_str = f"{profile['box_x']:.1f}x{profile['box_y']:.1f}x{profile['box_z']:.1f}"

            print(f"{profile['id']:<5} "
                  f"{profile['water_model']:<8} "
                  f"{ion_str:<20} "
                  f"{box_str:<25} "
                  f"{profile['supercell_scale']:<8.3f} "
                  f"{profile['simulation_time_ps']:<10.1f}")

        print(f"{'-'*80}\n")
        print(f"Total profiles: {len(profiles)}")


def cmd_info(args):
    """Show detailed information for a specific profile."""
    with SaxsDatabase(args.db) as db:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM saxs_profiles WHERE id = ?", (args.profile_id,))
        row = cursor.fetchone()

        if not row:
            print(f"Profile ID {args.profile_id} not found.", file=sys.stderr)
            return 1

        profile = db._row_to_dict(row)

        print(f"\n{'='*60}")
        print(f"SAXS Profile Details (ID: {profile['id']})")
        print(f"{'='*60}")

        print(f"\n--- Solvent Information ---")
        print(f"Water Model: {profile['water_model']}")
        print(f"Water Molecules: {profile['n_water_molecules']}")

        if profile['ion_counts']:
            print(f"\nIon Composition:")
            for ion, count in profile['ion_counts'].items():
                if count > 0:
                    charge_symbol = ""
                    if ion in ['Na', 'K']:
                        charge_symbol = "⁺"
                    elif ion == 'Cl':
                        charge_symbol = "⁻"
                    elif ion in ['Ca', 'Mg']:
                        charge_symbol = "²⁺"
                    print(f"  {ion}{charge_symbol}: {count}")

        print(f"\n--- Box Dimensions ---")
        print(f"Size: {profile['box_x']:.3f} x {profile['box_y']:.3f} x {profile['box_z']:.3f} Å")
        print(f"Volume: {profile['box_volume']:.2f} Å³")

        print(f"\n--- Supercell ---")
        print(f"Scale Factor: {profile['supercell_scale']:.4f}")
        print(f"Supercell Volume: {profile['supercell_volume']:.2f} Å³")

        print(f"\n--- Simulation Parameters ---")
        print(f"Simulation Time: {profile['simulation_time_ps']:.2f} ps")
        print(f"Frames Analyzed: {profile['n_frames_analyzed']}")
        print(f"Frame Stride: {profile['frame_stride']}")

        print(f"\n--- SAXS Calculation ---")
        print(f"Grid Size: {profile['grid_size']}")
        print(f"B-Spline Order: {profile['order']}")
        print(f"Bin Size: {profile['bin_size']}")
        print(f"Q Cutoff: {profile['qcut']}")

        print(f"\n--- System Properties ---")
        print(f"Total Atoms: {profile['n_atoms']}")
        if profile['density_g_cm3']:
            print(f"Density: {profile['density_g_cm3']:.4f} g/cm³")

        print(f"\n--- Profile Data ---")
        print(f"Data Points: {len(profile['profile_data'])}")
        if profile['profile_data']:
            q_min = profile['profile_data'][0][0]
            q_max = profile['profile_data'][-1][0]
            print(f"Q Range: {q_min:.6f} - {q_max:.6f} Å⁻¹")

        print(f"\n--- Metadata ---")
        print(f"Created: {profile['created_timestamp']}")
        print(f"Profile Hash: {profile['profile_hash'][:16]}...")
        if profile['notes']:
            print(f"Notes: {profile['notes']}")

        print(f"\n{'='*60}\n")


def cmd_export(args):
    """Export profile to CSV file."""
    with SaxsDatabase(args.db) as db:
        db.export_profile_csv(args.profile_id, args.output)
        print(f"Profile {args.profile_id} exported to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage SAXS profile database"
    )

    default_db = str(SaxsDefaults.get_default_database_path())
    parser.add_argument(
        "--db",
        default=default_db,
        help=f"Database file path (default: {default_db})"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all profiles")
    list_parser.add_argument(
        "--water-model",
        dest="water_model",
        help="Filter by water model (TIP3P, TIP4P, SPC, SPCE)"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed profile information")
    info_parser.add_argument("profile_id", type=int, help="Profile ID")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export profile to CSV")
    export_parser.add_argument("profile_id", type=int, help="Profile ID")
    export_parser.add_argument("output", help="Output CSV file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "export":
        cmd_export(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
