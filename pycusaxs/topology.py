"""
Topology module for molecular trajectory analysis using MDAnalysis.

This module provides the Topology class for analyzing GROMACS TPR/XTC files,
building molecular graphs, classifying molecules, and efficiently streaming
trajectory data for SAXS calculations.

Classes:
    Topology: Main class for topology and trajectory analysis

Key Methods:
    __init__(tpr_file, xtc_file): Initialize with topology and trajectory files
    iter_frames_stream(start, stop, step): Memory-efficient frame iteration
    get_atom_index(): Get atom indices grouped by element type
    count_molecules(): Count and categorize molecules
    read_frame(frame_number): Load a specific trajectory frame
"""
#!/usr/bin/env python
import argparse
import sys
from typing import Dict, List, Tuple, Iterator, Optional

import MDAnalysis as mda
import networkx as nx
import numpy as np
import warnings

# Unit conversion constant
NM_TO_ANGSTROM = 10.0  # Convert nanometers to Angstroms


class Topology:
    """
    Analyzes molecular topology and trajectory from GROMACS files.

    This class builds a graph representation of molecular connectivity,
    classifies molecules by type, and provides efficient access to
    trajectory data.

    Attributes:
        tpr_file (str): Path to GROMACS TPR topology file
        universe (MDAnalysis.Universe): MDAnalysis universe object
        molecules (list): All molecules as sets of atom indices
        protein_molecules (list): Protein molecules
        water_molecules (list): Water molecules
        ion_molecules (list): Ion molecules
        other_molecules (list): Other molecule types
    """

    def __init__(self, tpr_file: str, xtc_file: str):
        """
        Initialize Topology with GROMACS TPR and XTC files.

        Args:
            tpr_file: Path to GROMACS topology (.tpr) file
            xtc_file: Path to GROMACS trajectory (.xtc) file
        """
        warnings.filterwarnings(
            "ignore", message="No coordinate reader found for", category=UserWarning)
        self.tpr_file = tpr_file
        self.universe = mda.Universe(tpr_file, xtc_file)
        self.G = nx.Graph()
        self.molecules = []
        self.protein_molecules = []
        self.water_molecules = []
        self.ion_molecules = []
        self.other_molecules = []
        self.ts = None  # Current timestep (set by read_frame)
        self.__build_graph()
        self.__classify_molecules()
        print(f"Initialized Topology: {self.n_atoms} atoms, {self.n_frames} frames")

    def __build_graph(self):
        """
        Builds a graph where atoms are nodes and bonds are edges.
        """
        add_edge = self.G.add_edge  # Local function reference for speed

        # Add edges based on bonds
        for bond in self.universe.bonds:
            add_edge(bond.atoms[0].index, bond.atoms[1].index)

        # Find all connected components (molecules)
        self.molecules = list(nx.connected_components(self.G))

    def __classify_molecules(self):
        """
        Classifies the molecules into proteins, waters, ions, and others.
        """
        # Select residues for different types of molecules
        protein_residues = set(self.universe.select_atoms('protein').residues)
        water_residues = set(self.universe.select_atoms(
            'resname TIP3 TIP4 SOL WAT').residues)
        ion_residues = set(self.universe.select_atoms(
            'resname NA CL K CA MG').residues)

        # Classify each molecule based on its residues
        for molecule in self.molecules:
            molecule_residues = set(
                self.universe.atoms[list(molecule)].residues)
            if molecule_residues & protein_residues:
                self.protein_molecules.append(molecule)
            elif molecule_residues & water_residues:
                self.water_molecules.append(molecule)
            elif molecule_residues & ion_residues:
                self.ion_molecules.append(molecule)
            else:
                self.other_molecules.append(molecule)
        if self.protein_molecules:
            indices = set().union(*self.protein_molecules)
            self.protein_atoms = self.universe.atoms[list(indices)]
        else:
            self.protein_atoms = self.universe.atoms

    def count_molecules(self) -> Tuple[int, int, int, int, int]:
        """
        Count total molecules and categorize them by type.

        Returns:
            Tuple of (total_molecules, proteins, waters, ions, others)
        """
        num_molecules = len(self.molecules)
        num_proteins = len(self.protein_molecules)
        num_waters = len(self.water_molecules)
        num_ions = len(self.ion_molecules)
        num_others = len(self.other_molecules)

        return num_molecules, num_proteins, num_waters, num_ions, num_others

    def generate_molecule_dict(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Generate dictionary with detailed molecule information.

        Returns:
            Dictionary with keys 'proteins', 'waters', 'ions', 'others',
            each containing numbered molecule entries with atom indices
        """
        molecule_dict = {
            'proteins': {},
            'waters': {},
            'ions': {},
            'others': {}
        }

        # Populate the dictionary with molecules
        for i, molecule in enumerate(self.protein_molecules):
            molecule_dict['proteins'][f'{i}'] = list(molecule)

        for i, molecule in enumerate(self.water_molecules):
            molecule_dict['waters'][f'{i}'] = list(molecule)

        for i, molecule in enumerate(self.ion_molecules):
            molecule_dict['ions'][f'{i}'] = list(molecule)

        for i, molecule in enumerate(self.other_molecules):
            molecule_dict['others'][f'{i}'] = list(molecule)

        return molecule_dict

    def get_atom_index(self) -> Dict[str, List[int]]:
        """
        Get atom indices grouped by element type.

        Returns:
            Dictionary mapping element symbols to lists of atom indices
        """
        atom_index = {}
        for atom in self.universe.atoms:
            atom_type = atom.element
            if atom_type not in atom_index:
                atom_index[atom_type] = []
            atom_index[atom_type].append(atom.index)
        return atom_index

    def myUniverse(self) -> mda.Universe:
        """
        Get the MDAnalysis Universe object.

        Returns:
            The underlying MDAnalysis Universe
        """
        return self.universe

    def read_frame(self, frame_number: int) -> mda.coordinates.timestep.Timestep:
        """
        Read a specific frame from the trajectory.

        Args:
            frame_number: Frame index to read (0-based)

        Returns:
            Timestep object for the requested frame

        Raises:
            IndexError: If frame_number is out of bounds
        """
        if not 0 <= frame_number < self.n_frames:
            raise IndexError(
                f"Frame {frame_number} out of range [0, {self.n_frames})")
        self.ts = self.universe.trajectory[frame_number]
        return self.ts

    def read_traj(self, frame_start: int, frame_end: int):
        """
        Read a range of frames (deprecated - use iter_frames_stream instead).

        Args:
            frame_start: Starting frame index
            frame_end: Ending frame index
        """
        self.frames = self.universe.trajectory[frame_start:frame_end]

    def get_box(self) -> np.ndarray:
        """
        Get box dimensions for current frame in Angstroms.

        Returns:
            3x3 array of triclinic box dimensions (Angstroms)

        Raises:
            RuntimeError: If read_frame() hasn't been called
        """
        if self.ts is None:
            raise RuntimeError("Call read_frame() before get_box()")
        return self.ts.triclinic_dimensions / NM_TO_ANGSTROM

    def get_step(self) -> int:
        """
        Get step number for current frame.

        Returns:
            Frame number/index

        Raises:
            RuntimeError: If read_frame() hasn't been called
        """
        if self.ts is None:
            raise RuntimeError("Call read_frame() before get_step()")
        return self.ts.frame

    def get_time(self) -> float:
        """
        Get simulation time for current frame.

        Returns:
            Time in picoseconds

        Raises:
            RuntimeError: If read_frame() hasn't been called
        """
        if self.ts is None:
            raise RuntimeError("Call read_frame() before get_time()")
        return self.ts.time

    def get_coordinates(self) -> np.ndarray:
        """
        Get atomic coordinates for current frame in Angstroms.

        Returns:
            Nx3 array of atom positions (Angstroms)

        Raises:
            RuntimeError: If read_frame() hasn't been called
        """
        if self.ts is None:
            raise RuntimeError("Call read_frame() before get_coordinates()")
        return self.ts.positions / NM_TO_ANGSTROM

    def iter_frames_stream(self, start: int, stop: int, step: int = 1) -> Iterator[Dict]:
        """
        Memory-efficient streaming iterator for large trajectories.

        This method is optimized for processing very large trajectory files
        (>20GB) by yielding one frame at a time without loading all frames
        into memory. Ideal for SAXS calculations on extensive MD simulations.

        Args:
            start: Starting frame index (inclusive)
            stop: Stopping frame index (exclusive)
            step: Frame stride (default: 1, every frame)

        Yields:
            dict: Frame data with keys:
                - 'frame' (int): Frame number
                - 'time' (float): Simulation time (ps)
                - 'positions' (ndarray): Atom positions in Angstroms (Nx3)
                - 'box' (ndarray): Box dimensions in Angstroms (3x3)

        Example:
            >>> for frame_data in topology.iter_frames_stream(0, 1000, 10):
            ...     coords = frame_data['positions']
            ...     # Process coords for SAXS calculation
        """
        for ts in self.universe.trajectory[start:stop:step]:
            yield {
                'frame': ts.frame,
                'time': ts.time,
                'positions': self.universe.atoms.positions / NM_TO_ANGSTROM,
                'box': ts.triclinic_dimensions / NM_TO_ANGSTROM
            }

    @property
    def n_frames(self) -> int:
        """Total number of frames in trajectory."""
        return len(self.universe.trajectory)

    @property
    def n_atoms(self) -> int:
        """Total number of atoms in the system."""
        return len(self.universe.atoms)

    def __repr__(self) -> str:
        """String representation of Topology object."""
        return (f"Topology(tpr='{self.tpr_file}', "
                f"n_atoms={self.n_atoms}, n_frames={self.n_frames})")


def main():
    """
    Command-line interface for topology analysis and molecule counting.

    This function demonstrates basic usage of the Topology class for
    analyzing molecular systems from GROMACS files.
    """
    parser = argparse.ArgumentParser(
        description='Analyze TPR and XTC files and count molecules in the system.')
    parser.add_argument('-s', '--tpr', type=str, required=True,
                        help='Path to the GROMACS TPR topology file')
    parser.add_argument('-x', '--xtc', type=str, required=True,
                        help='Path to the GROMACS XTC trajectory file')
    parser.add_argument('-f', '--frame', type=int, default=0,
                        help='Frame number to analyze (default: 0)')

    args = parser.parse_args()

    # Initialize Topology and analyze system
    print(f"Loading topology from: {args.tpr}")
    print(f"Loading trajectory from: {args.xtc}")
    top = Topology(args.tpr, args.xtc)

    # Count and categorize molecules
    (num_molecules, num_proteins, num_waters,
     num_ions, num_others) = top.count_molecules()

    print(f"\nMolecule Statistics:")
    print(f"  Total molecules: {num_molecules}")
    print(f"  Proteins: {num_proteins}")
    print(f"  Waters: {num_waters}")
    print(f"  Ions: {num_ions}")
    if num_others:
        print(f"  Others: {num_others}")

    # Get atom indices by element
    atom_index = top.get_atom_index()
    print(f"\nElement Distribution:")
    for element, indices in sorted(atom_index.items()):
        print(f"  {element}: {len(indices)} atoms")

    # Analyze specific frame
    print(f"\nAnalyzing frame {args.frame}:")
    top.read_frame(args.frame)
    box = top.get_box()
    coords = top.get_coordinates()
    print(f"  Box dimensions (Å): {box[0, 0]:.2f} × {box[1, 1]:.2f} × {box[2, 2]:.2f}")
    print(f"  Time: {top.get_time():.2f} ps")
    print(f"  Number of atoms: {len(coords)}")

    # Example: Stream through first 10 frames
    print(f"\nStreaming example (first 10 frames):")
    for frame_data in top.iter_frames_stream(0, min(10, top.n_frames), 1):
        print(f"  Frame {frame_data['frame']}: "
              f"{len(frame_data['positions'])} atoms at {frame_data['time']:.2f} ps")


if __name__ == "__main__":
    main()
