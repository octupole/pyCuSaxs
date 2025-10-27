#!/usr/bin/env python3
"""
Command-line tool for managing SAXS profile database.

Usage:
    saxs-db [--db DATABASE | --use-reference] list [--water-model MODEL]
    saxs-db [--db DATABASE | --use-reference] info ID
    saxs-db [--db DATABASE | --use-reference] export ID OUTPUT.csv
    saxs-db [--db DATABASE | --use-reference] plot ID OUTPUT.dat
    saxs-db [--db DATABASE] delete ID [ID ...] [-y]

Options:
    --db DATABASE        Use custom database file path
    --use-reference      Use the reference solvent database (read-only)
                        Located at: <env>/lib/python3.x/site-packages/pycusaxs/data/reference_solvents.db

Note: The reference database contains validated reference solvent SAXS profiles.
      Deletion is not allowed from the reference database.
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

        print(f"\n{'='*130}")
        print(f"SAXS Profiles in {args.db}")
        print(f"{'='*130}")
        print(f"{'ID':<5} {'Water':<8} {'Ions':<15} {'Other Molecules':<20} {'Box (Å)':<20} {'Grid':<15} {'Supercell':<15} {'Time (ps)':<10} {'Density (g/cm³)':<18}")
        print(f"{'-'*130}")

        for profile in profiles:
            # Format ions
            ion_str = ', '.join([f"{k}:{v}" for k, v in profile['ion_counts'].items() if v > 0])
            if not ion_str:
                ion_str = "none"

            # Format other molecules (proteins, ligands)
            other_mols = profile.get('other_molecules', {})
            if other_mols:
                mol_str = ', '.join([f"{k}:{v}" for k, v in other_mols.items()])
            else:
                mol_str = "none"

            box_str = f"{profile['box_x']:.1f}x{profile['box_y']:.1f}x{profile['box_z']:.1f}"

            # Format grid (cell grid)
            grid = profile['grid_size']
            grid_str = f"{grid[0]}x{grid[1]}x{grid[2]}"

            # Calculate supercell grid
            scale = profile['supercell_scale']
            supercell_str = f"{int(grid[0]*scale)}x{int(grid[1]*scale)}x{int(grid[2]*scale)}"
            density = profile.get('density_g_cm3')
            density_str = f"{density:.4f}" if density is not None else "N/A"

            print(f"{profile['id']:<5} "
                  f"{profile['water_model']:<8} "
                  f"{ion_str:<15} "
                  f"{mol_str:<20} "
                  f"{box_str:<20} "
                  f"{grid_str:<15} "
                  f"{supercell_str:<15} "
                  f"{profile['simulation_time_ps']:<10.1f} "
                  f"{density_str:<18}")

        print(f"{'-'*130}\n")
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

        other_mols = profile.get('other_molecules', {})
        if other_mols:
            print(f"\nOther Molecules (proteins, ligands, etc.):")
            for mol, count in other_mols.items():
                print(f"  {mol}: {count}")

        print(f"\n--- Box Dimensions ---")
        print(f"Size: {profile['box_x']:.3f} x {profile['box_y']:.3f} x {profile['box_z']:.3f} Å")
        print(f"Volume: {profile['box_volume']:.2f} Å³")

        print(f"\n--- Supercell ---")
        print(f"Scale Factor: {profile['supercell_scale']:.4f}")
        grid = profile['grid_size']
        scale = profile['supercell_scale']
        print(f"Supercell Grid: {int(grid[0]*scale)} x {int(grid[1]*scale)} x {int(grid[2]*scale)}")
        print(f"Supercell Volume: {profile['supercell_volume']:.2f} Å³")

        print(f"\n--- Simulation Parameters ---")
        print(f"Simulation Time: {profile['simulation_time_ps']:.2f} ps")
        print(f"Frames Analyzed: {profile['n_frames_analyzed']}")
        print(f"Frame Stride: {profile['frame_stride']}")

        print(f"\n--- SAXS Calculation ---")
        print(f"Grid Size: {profile['grid_size']}")
        print(f"B-Spline Order: {profile['spline_order']}")
        print(f"Bin Size: {profile['bin_size']}")
        print(f"Q Cutoff: {profile['qcut']}")

        print(f"\n--- System Properties ---")
        print(f"Total Atoms: {profile['n_atoms']}")
        if profile.get('density_g_cm3') is not None:
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


def cmd_plot(args):
    """Export profile to simple x-y format for plotting (xmgrace, gnuplot, etc.)."""
    with SaxsDatabase(args.db) as db:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM saxs_profiles WHERE id = ?", (args.profile_id,))
        row = cursor.fetchone()

        if not row:
            print(f"Profile ID {args.profile_id} not found.", file=sys.stderr)
            return 1

        profile = db._row_to_dict(row)

        with open(args.output, 'w') as f:
            # Write header with metadata as comments
            f.write(f"# SAXS Profile from Database\n")
            f.write(f"# Profile ID: {profile['id']}\n")
            f.write(f"# Water Model: {profile['water_model']}\n")
            f.write(f"# Box Size: {profile['box_x']:.3f} x {profile['box_y']:.3f} x {profile['box_z']:.3f} Å\n")
            f.write(f"# Grid: {profile['grid_size']}\n")
            f.write(f"# Supercell Scale: {profile['supercell_scale']:.4f}\n")
            f.write(f"# Simulation Time: {profile['simulation_time_ps']:.2f} ps\n")
            density = profile.get('density_g_cm3')
            density_str = f"{density:.4f}" if density is not None else "N/A"
            f.write(f"# Density: {density_str} g/cm³\n")
            f.write(f"#\n")
            f.write(f"# q (Å⁻¹)    I(q)\n")

            # Write data in simple x-y format
            for q, iq in profile['profile_data']:
                f.write(f"{q:.6f}  {iq:.6e}\n")

        print(f"Profile {args.profile_id} exported to {args.output} (xmgrace format)")


def cmd_delete(args):
    """Delete one or more profiles from the database."""
    # Safety check: prevent deletion from reference database
    ref_db_path = str(SaxsDefaults.get_reference_database_path())
    if args.db == ref_db_path:
        print("ERROR: Cannot delete profiles from reference database.", file=sys.stderr)
        print("The reference database is read-only and contains reference solvent profiles.", file=sys.stderr)
        return 1

    with SaxsDatabase(args.db) as db:
        cursor = db.conn.cursor()

        # Show what will be deleted
        for profile_id in args.profile_ids:
            cursor.execute("SELECT * FROM saxs_profiles WHERE id = ?", (profile_id,))
            row = cursor.fetchone()

            if not row:
                print(f"Profile ID {profile_id} not found.", file=sys.stderr)
                continue

            profile = db._row_to_dict(row)
            print(f"\nProfile {profile_id}:")
            print(f"  Water Model: {profile['water_model']}")
            print(f"  Grid: {profile['grid_size']}")
            print(f"  Supercell Scale: {profile['supercell_scale']:.4f}")
            print(f"  Simulation Time: {profile['simulation_time_ps']:.2f} ps")
            print(f"  Created: {profile['created_timestamp']}")

        # Confirm deletion
        if not args.yes:
            response = input(f"\nDelete {len(args.profile_ids)} profile(s)? [y/N]: ")
            if response.lower() != 'y':
                print("Deletion cancelled.")
                return 0

        # Delete profiles
        deleted = 0
        for profile_id in args.profile_ids:
            cursor.execute("DELETE FROM saxs_profiles WHERE id = ?", (profile_id,))
            if cursor.rowcount > 0:
                deleted += 1

        db.conn.commit()
        print(f"\nDeleted {deleted} profile(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Manage SAXS profile database"
    )

    default_db = str(SaxsDefaults.get_user_database_path())
    ref_db = str(SaxsDefaults.get_reference_database_path())

    parser.add_argument(
        "--db",
        default=default_db,
        help=f"Database file path (default: {default_db})"
    )
    parser.add_argument(
        "--use-reference",
        action="store_true",
        help=f"Use reference solvent database instead of user database (located at {ref_db})"
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

    # Plot command (xmgrace format)
    plot_parser = subparsers.add_parser("plot", help="Export profile to x-y format for plotting")
    plot_parser.add_argument("profile_id", type=int, help="Profile ID")
    plot_parser.add_argument("output", help="Output file path (.dat, .xy, etc.)")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete profile(s) from database")
    delete_parser.add_argument("profile_ids", type=int, nargs='+', metavar="ID", help="Profile ID(s) to delete")
    delete_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Override database path if --use-reference is specified
    if args.use_reference:
        args.db = ref_db
        print(f"Using reference database: {args.db}\n")

    if args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "plot":
        return cmd_plot(args)
    elif args.command == "delete":
        return cmd_delete(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
