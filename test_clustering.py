#!/usr/bin/env python
"""Test script for cluster reconstruction with SDS molecules."""

from pycusaxs.topology import Topology
import MDAnalysis as mda
from pathlib import Path

def test_clustering(tpr_file, xtc_file, selection='resname SDS', cutoff=15.0,
                   n_frames=5, output_dir='cluster_output'):
    """
    Test cluster reconstruction on a trajectory.

    Args:
        tpr_file: Path to TPR topology file
        xtc_file: Path to XTC trajectory file
        selection: Atom selection for clustering (default: 'resname SDS')
        cutoff: DBSCAN clustering cutoff in Angstroms (default: 15.0)
        n_frames: Number of frames to process (default: 5)
        output_dir: Directory to save PDB files (default: 'cluster_output')
    """
    print(f"Loading topology: {tpr_file}")
    print(f"Loading trajectory: {xtc_file}")
    top = Topology(tpr_file, xtc_file)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")

    print(f"\nProcessing {n_frames} frames with cluster reconstruction...")
    print(f"Selection: {selection}")
    print(f"Cutoff: {cutoff} Å")
    print("-" * 60)

    frame_count = 0
    for frame_data in top.iter_frames_stream(
        0, n_frames, 1,
        cluster_selection=selection,
        cluster_cutoff=cutoff
    ):
        frame_num = frame_data['frame']
        frame_count += 1

        # Write PDB file for this frame
        pdb_file = output_path / f"frame_{frame_num:05d}_clustered.pdb"

        # Update universe positions with the clustered/reconstructed coordinates
        top.universe.atoms.positions = frame_data['positions']

        # Write selection to PDB
        selection_atoms = top.universe.select_atoms(selection)
        selection_atoms.write(str(pdb_file))
        print(f"  Wrote {pdb_file}")

    print("-" * 60)
    print(f"\nSuccessfully processed {frame_count} frames!")
    print(f"PDB files saved in: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test cluster reconstruction')
    parser.add_argument('-s', '--tpr', required=True, help='TPR topology file')
    parser.add_argument('-f', '--xtc', required=True, help='XTC trajectory file')
    parser.add_argument('--selection', default='resname SDS',
                        help='Atom selection (default: resname SDS)')
    parser.add_argument('--cutoff', type=float, default=15.0,
                        help='DBSCAN cutoff in Angstroms (default: 15.0)')
    parser.add_argument('-n', '--nframes', type=int, default=5,
                        help='Number of frames to process (default: 5)')
    parser.add_argument('-o', '--output', default='cluster_output',
                        help='Output directory for PDB files (default: cluster_output)')

    args = parser.parse_args()

    test_clustering(args.tpr, args.xtc, args.selection, args.cutoff,
                   args.nframes, args.output)
