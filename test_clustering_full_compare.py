#!/usr/bin/env python
"""Compare original vs clustered/reconstructed FULL system coordinates."""

from pycusaxs.topology import Topology
import MDAnalysis as mda
from pathlib import Path

def compare_full_system(tpr_file, xtc_file, cluster_selection='resname SDS',
                       cutoff=15.0, frame_num=0, output_dir='compare_full_output'):
    """
    Compare original vs reconstructed coordinates for FULL SYSTEM.

    This shows the complete system including all water, ions, and molecules
    - exactly what gets passed to RunSaxs.

    Args:
        tpr_file: Path to TPR topology file
        xtc_file: Path to XTC trajectory file
        cluster_selection: Selection for clustering (default: 'resname SDS')
        cutoff: DBSCAN clustering cutoff in Angstroms (default: 15.0)
        frame_num: Frame number to compare (default: 0)
        output_dir: Directory to save PDB files (default: 'compare_full_output')
    """
    print(f"Loading topology: {tpr_file}")
    print(f"Loading trajectory: {xtc_file}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")

    # Load original frame
    print(f"\n1. Saving ORIGINAL frame {frame_num} (before clustering)...")
    print(f"   FULL SYSTEM - all atoms as read from trajectory")
    top_original = Topology(tpr_file, xtc_file)
    top_original.read_frame(frame_num)

    original_file = output_path / f"frame_{frame_num:05d}_ORIGINAL_full.pdb"
    top_original.universe.atoms.write(str(original_file))
    print(f"   Wrote {original_file} ({len(top_original.universe.atoms)} atoms)")

    # Load and process with clustering
    print(f"\n2. Processing frame {frame_num} with CLUSTER RECONSTRUCTION...")
    print(f"   Clustering selection: {cluster_selection}")
    print(f"   Cutoff: {cutoff} Å")
    print(f"   Output: FULL SYSTEM - all atoms with reconstructed clusters")
    top_clustered = Topology(tpr_file, xtc_file)

    for frame_data in top_clustered.iter_frames_stream(
        frame_num, frame_num + 1, 1,
        cluster_selection=cluster_selection,
        cluster_cutoff=cutoff
    ):
        # Update universe with reconstructed coordinates (full system)
        top_clustered.universe.atoms.positions = frame_data['positions']

        # Write reconstructed FULL SYSTEM to PDB
        clustered_file = output_path / f"frame_{frame_num:05d}_RECONSTRUCTED_full.pdb"
        top_clustered.universe.atoms.write(str(clustered_file))
        print(f"   Wrote {clustered_file} ({len(top_clustered.universe.atoms)} atoms)")

    print(f"\n" + "="*70)
    print(f"COMPARISON FILES CREATED (FULL SYSTEM):")
    print(f"  Original:      {original_file}")
    print(f"  Reconstructed: {clustered_file}")

    # Show system composition
    info = top_clustered.get_system_info()
    print(f"\nSystem composition:")
    print(f"  - Total atoms: {info['n_atoms']}")
    if 'n_water_molecules' in info:
        print(f"  - Water molecules: {info['n_water_molecules']}")
    if 'ion_counts' in info:
        for ion, count in info['ion_counts'].items():
            if count > 0:
                print(f"  - {ion} ions: {count}")

    # Count selection atoms
    sel_atoms = top_clustered.universe.select_atoms(cluster_selection)
    print(f"  - Clustered ({cluster_selection}): {len(sel_atoms)} atoms")

    print(f"\nLoad both files in VMD/PyMOL to visualize the difference!")
    print(f"Focus on '{cluster_selection}' molecules to see reconstruction.")
    print(f"Water and other molecules remain in their original positions.")
    print("="*70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare original vs cluster-reconstructed FULL system')
    parser.add_argument('-s', '--tpr', required=True, help='TPR topology file')
    parser.add_argument('-f', '--xtc', required=True, help='XTC trajectory file')
    parser.add_argument('--selection', default='resname SDS',
                        help='Selection for clustering (default: resname SDS)')
    parser.add_argument('--cutoff', type=float, default=15.0,
                        help='DBSCAN cutoff in Angstroms (default: 15.0)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame number to compare (default: 0)')
    parser.add_argument('-o', '--output', default='compare_full_output',
                        help='Output directory (default: compare_full_output)')

    args = parser.parse_args()

    compare_full_system(args.tpr, args.xtc, args.selection, args.cutoff,
                       args.frame, args.output)
