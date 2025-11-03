#!/usr/bin/env python
"""Test cluster reconstruction with full system output (including water)."""

from pycusaxs.topology import Topology
import MDAnalysis as mda
from pathlib import Path

def test_clustering_full(tpr_file, xtc_file, cluster_selection='resname SDS',
                         cutoff=15.0, n_frames=5, output_dir='cluster_full_output'):
    """
    Test cluster reconstruction and output FULL system coordinates (all atoms).

    This outputs the complete system including water, ions, and all molecules
    - exactly what gets passed to RunSaxs for SAXS calculation.

    Args:
        tpr_file: Path to TPR topology file
        xtc_file: Path to XTC trajectory file
        cluster_selection: Selection string for atoms to cluster and reconstruct
                          (default: 'resname SDS')
        cutoff: DBSCAN clustering cutoff in Angstroms (default: 15.0)
        n_frames: Number of frames to process (default: 5)
        output_dir: Directory to save PDB files (default: 'cluster_full_output')
    """
    print(f"Loading topology: {tpr_file}")
    print(f"Loading trajectory: {xtc_file}")
    top = Topology(tpr_file, xtc_file)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")

    print(f"\nProcessing {n_frames} frames with cluster reconstruction...")
    print(f"Cluster selection: {cluster_selection}")
    print(f"Cutoff: {cutoff} Å")
    print(f"Output: FULL SYSTEM (all {len(top.universe.atoms)} atoms)")
    print("-" * 70)

    frame_count = 0
    for frame_data in top.iter_frames_stream(
        0, n_frames, 1,
        cluster_selection=cluster_selection,
        cluster_cutoff=cutoff
    ):
        frame_num = frame_data['frame']
        frame_count += 1

        # Write FULL SYSTEM PDB file for this frame
        # This is exactly what gets passed to RunSaxs
        pdb_file = output_path / f"frame_{frame_num:05d}_full_system.pdb"

        # Update universe positions with the clustered/reconstructed coordinates
        top.universe.atoms.positions = frame_data['positions']

        # Write ALL atoms to PDB (entire system)
        top.universe.atoms.write(str(pdb_file))
        print(f"  Wrote {pdb_file} ({len(top.universe.atoms)} atoms)")

    print("-" * 70)
    print(f"\nSuccessfully processed {frame_count} frames!")
    print(f"PDB files (full system) saved in: {output_path}")
    print(f"\nThese PDB files contain:")

    # Show what's in the system
    info = top.get_system_info()
    print(f"  - Total atoms: {info['n_atoms']}")
    if 'n_water_molecules' in info:
        print(f"  - Water molecules: {info['n_water_molecules']}")
    if 'ion_counts' in info:
        for ion, count in info['ion_counts'].items():
            print(f"  - {ion} ions: {count}")
    print(f"  - And all other molecules in the system")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Test cluster reconstruction with full system output')
    parser.add_argument('-s', '--tpr', required=True, help='TPR topology file')
    parser.add_argument('-f', '--xtc', required=True, help='XTC trajectory file')
    parser.add_argument('--selection', default='resname SDS',
                        help='Selection for clustering (default: resname SDS)')
    parser.add_argument('--cutoff', type=float, default=15.0,
                        help='DBSCAN cutoff in Angstroms (default: 15.0)')
    parser.add_argument('-n', '--nframes', type=int, default=5,
                        help='Number of frames to process (default: 5)')
    parser.add_argument('-o', '--output', default='cluster_full_output',
                        help='Output directory for PDB files (default: cluster_full_output)')

    args = parser.parse_args()

    test_clustering_full(args.tpr, args.xtc, args.selection, args.cutoff,
                        args.nframes, args.output)
