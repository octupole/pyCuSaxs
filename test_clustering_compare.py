#!/usr/bin/env python
"""Compare original vs clustered/reconstructed structures."""

from pycusaxs.topology import Topology
import MDAnalysis as mda
from pathlib import Path

def compare_clustering(tpr_file, xtc_file, selection='resname SDS', cutoff=15.0,
                       frame_num=0, output_dir='compare_output'):
    """
    Compare original vs reconstructed coordinates for a single frame.

    Args:
        tpr_file: Path to TPR topology file
        xtc_file: Path to XTC trajectory file
        selection: Atom selection for clustering (default: 'resname SDS')
        cutoff: DBSCAN clustering cutoff in Angstroms (default: 15.0)
        frame_num: Frame number to compare (default: 0)
        output_dir: Directory to save PDB files (default: 'compare_output')
    """
    print(f"Loading topology: {tpr_file}")
    print(f"Loading trajectory: {xtc_file}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")

    # Load original frame
    print(f"\n1. Saving ORIGINAL frame {frame_num} (before clustering)...")
    top_original = Topology(tpr_file, xtc_file)
    top_original.read_frame(frame_num)
    original_atoms = top_original.universe.select_atoms(selection)
    original_file = output_path / f"frame_{frame_num:05d}_ORIGINAL.pdb"
    original_atoms.write(str(original_file))
    print(f"   Wrote {original_file}")

    # Load and process with clustering
    print(f"\n2. Processing frame {frame_num} with CLUSTER RECONSTRUCTION...")
    print(f"   Selection: {selection}")
    print(f"   Cutoff: {cutoff} Å")
    top_clustered = Topology(tpr_file, xtc_file)

    for frame_data in top_clustered.iter_frames_stream(
        frame_num, frame_num + 1, 1,
        cluster_selection=selection,
        cluster_cutoff=cutoff
    ):
        # Update universe with reconstructed coordinates
        top_clustered.universe.atoms.positions = frame_data['positions']

        # Write reconstructed selection to PDB
        clustered_file = output_path / f"frame_{frame_num:05d}_RECONSTRUCTED.pdb"
        clustered_atoms = top_clustered.universe.select_atoms(selection)
        clustered_atoms.write(str(clustered_file))
        print(f"   Wrote {clustered_file}")

    print(f"\n" + "="*70)
    print(f"COMPARISON FILES CREATED:")
    print(f"  Original:      {original_file}")
    print(f"  Reconstructed: {clustered_file}")
    print(f"\nLoad both files in VMD/PyMOL to visualize the difference!")
    print(f"Look for SDS molecules that were split across periodic boundaries")
    print(f"- they should now be whole in the reconstructed file.")
    print("="*70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare original vs cluster-reconstructed structures')
    parser.add_argument('-s', '--tpr', required=True, help='TPR topology file')
    parser.add_argument('-f', '--xtc', required=True, help='XTC trajectory file')
    parser.add_argument('--selection', default='resname SDS',
                        help='Atom selection (default: resname SDS)')
    parser.add_argument('--cutoff', type=float, default=15.0,
                        help='DBSCAN cutoff in Angstroms (default: 15.0)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame number to compare (default: 0)')
    parser.add_argument('-o', '--output', default='compare_output',
                        help='Output directory (default: compare_output)')

    args = parser.parse_args()

    compare_clustering(args.tpr, args.xtc, args.selection, args.cutoff,
                       args.frame, args.output)
