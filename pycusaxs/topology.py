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
from __future__ import annotations

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
        print(
            f"Initialized Topology: {self.n_atoms} atoms, {self.n_frames} frames")

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

    def detect_water_model(self) -> str:
        """
        Detect the water model used in the simulation by analyzing charges and geometry.

        Distinguishes between:
        - SPC vs SPC/E (by charge: SPC has ±0.41e, SPC/E has ±0.4238e)
        - TIP3P vs TIP4P (by atom count: TIP4P has virtual site/extra atom)

        Returns:
            Water model name in uppercase (e.g., 'TIP3P', 'TIP4P', 'SPC', 'SPCE')
            or empty string if no water found. Returns uppercase to match C++ code expectations.
        """
        if not self.water_molecules:
            return ""

        # Get first water molecule to check
        water_atoms = self.universe.atoms[list(self.water_molecules[0])]
        resname = water_atoms.residues[0].resname.upper()

        # Quick mapping for explicitly named models
        if resname == 'TIP3':
            return 'TIP3P'
        elif resname == 'TIP4':
            return 'TIP4P'

        # For SOL and WAT, need to check charges/geometry to distinguish models
        if resname in ['SOL', 'WAT']:
            # Get charges if available
            try:
                if hasattr(water_atoms, 'charges'):
                    charges = water_atoms.charges
                    # Get oxygen charge (usually the first atom)
                    o_atoms = water_atoms.select_atoms('element O')
                    if len(o_atoms) > 0:
                        o_charge = abs(o_atoms[0].charge)

                        # SPC: O charge ≈ -0.82e, H charge ≈ +0.41e
                        # SPC/E: O charge ≈ -0.8476e, H charge ≈ +0.4238e
                        # TIP3P: O charge ≈ -0.834e, H charge ≈ +0.417e
                        # TIP4P: O charge ≈ -1.04e (or virtual site charge)

                        if o_charge > 0.845:  # |O charge| > 0.845
                            # Could be TIP4P or check number of atoms
                            if len(water_atoms) > 3:  # TIP4P has virtual site
                                return 'TIP4P'
                            else:
                                return 'SPCE'  # SPC/E has -0.8476
                        elif o_charge > 0.83:  # 0.83 < |O charge| < 0.845
                            return 'TIP3P'  # TIP3P has -0.834
                        elif o_charge > 0.825:  # 0.825 < |O charge| < 0.83
                            return 'SPC'  # SPC has -0.82
                        else:
                            # Check H charge to distinguish
                            h_atoms = water_atoms.select_atoms('element H')
                            if len(h_atoms) > 0:
                                h_charge = abs(h_atoms[0].charge)
                                if h_charge > 0.42:  # SPC/E: +0.4238
                                    return 'SPCE'
                                elif h_charge > 0.415:  # TIP3P: +0.417
                                    return 'TIP3P'
                                else:  # SPC: +0.41
                                    return 'SPC'
            except (AttributeError, IndexError):
                pass

            # Fallback: check number of atoms in water molecule
            if len(water_atoms) > 3:
                return 'TIP4P'  # TIP4P has 4 atoms (O, H, H, virtual site)

            # Default fallback based on residue name
            if resname == 'SOL':
                return 'SPC'  # GROMACS default
            else:
                return 'TIP3P'  # Generic water default

        # Unknown water model, return as-is
        return resname

    def count_ions(self) -> Dict[str, int]:
        """
        Count individual ion types in the system.

        Returns:
            Dictionary with ion counts: {'Na': count, 'Cl': count, 'K': count, 'Ca': count, 'Mg': count}
        """
        ion_counts = {'Na': 0, 'Cl': 0, 'K': 0, 'Ca': 0, 'Mg': 0}

        if not self.ion_molecules:
            return ion_counts

        # Get all ion atoms
        ion_indices = set().union(*self.ion_molecules)
        ion_atoms = self.universe.atoms[list(ion_indices)]

        # Count by residue name
        for residue in ion_atoms.residues:
            resname = residue.resname.upper()
            if resname in ['NA', 'SOD']:  # Sodium
                ion_counts['Na'] += 1
            elif resname in ['CL', 'CLA']:  # Chloride
                ion_counts['Cl'] += 1
            elif resname == 'K':  # Potassium
                ion_counts['K'] += 1
            elif resname in ['CA', 'CAL']:  # Calcium
                ion_counts['Ca'] += 1
            elif resname == 'MG':  # Magnesium
                ion_counts['Mg'] += 1

        return ion_counts

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

    def get_system_info(self) -> Dict:
        """
        Extract comprehensive system information for database storage.

        Returns:
            Dictionary containing detailed system information including:
            - Topology file information
            - Atom counts by element and residue
            - Molecule counts and compositions
            - Water model information
            - Ion composition
            - Box dimensions and volume
            - Trajectory information
            - Force field information (if available)
        """
        info = {}

        # Basic file information
        info['topology_file'] = self.tpr_file
        info['n_atoms'] = self.universe.atoms.n_atoms
        info['n_residues'] = self.universe.atoms.n_residues
        info['n_segments'] = self.universe.atoms.n_segments
        info['n_frames'] = self.n_frames

        # Trajectory information
        if self.universe.trajectory:
            info['trajectory_length_frames'] = len(self.universe.trajectory)
            info['dt'] = getattr(self.universe.trajectory, 'dt', None)
            info['total_time'] = getattr(
                self.universe.trajectory, 'totaltime', None)

        # Atom counts by element
        element_counts = {}
        for atom in self.universe.atoms:
            elem = atom.element
            element_counts[elem] = element_counts.get(elem, 0) + 1
        info['element_counts'] = element_counts

        # Molecule type counts
        info['n_proteins'] = len(self.protein_molecules)
        info['n_water_molecules'] = len(self.water_molecules)
        info['n_ion_molecules'] = len(self.ion_molecules)
        info['n_other_molecules'] = len(self.other_molecules)

        # Detailed water information
        if self.water_molecules:
            water_residues = self.universe.atoms[list(
                self.water_molecules[0])].residues
            if len(water_residues) > 0:
                info['water_resname'] = water_residues[0].resname
            info['detected_water_model'] = self.detect_water_model()

            # Count total water atoms
            total_water_atoms = sum(len(mol) for mol in self.water_molecules)
            info['n_water_atoms'] = total_water_atoms

        # Detailed ion information
        ion_counts = self.count_ions()

        # If count_ions returned empty, try to extract from residue_counts
        # (This handles cases where ion molecules weren't properly detected)
        if all(v == 0 for v in ion_counts.values()):
            # Will be populated from residue_counts below
            pass

        info['ion_counts'] = {k: v for k, v in ion_counts.items() if v > 0}

        # Residue composition
        residue_counts = {}
        for residue in self.universe.residues:
            resname = residue.resname
            residue_counts[resname] = residue_counts.get(resname, 0) + 1
        info['residue_counts'] = residue_counts

        # Extract ions from residue_counts if not already found
        if not info['ion_counts']:
            for resname, count in residue_counts.items():
                resname_upper = resname.upper()
                if resname_upper in ['NA', 'SOD']:
                    ion_counts['Na'] = count
                elif resname_upper in ['CL', 'CLA']:
                    ion_counts['Cl'] = count
                elif resname_upper == 'K':
                    ion_counts['K'] = count
                elif resname_upper in ['CA', 'CAL']:
                    ion_counts['Ca'] = count
                elif resname_upper == 'MG':
                    ion_counts['Mg'] = count
            info['ion_counts'] = {k: v for k, v in ion_counts.items() if v > 0}

        # Extract other molecules (non-water, non-ion)
        # These are proteins, ligands, lipids, etc.
        water_resnames = ['SOL', 'WAT', 'TIP3', 'TIP4', 'SPC', 'HOH', 'H2O']
        ion_resnames = ['NA', 'SOD', 'CL', 'CLA', 'K', 'CA', 'CAL', 'MG']
        other_molecules = {}
        for resname, count in residue_counts.items():
            if resname.upper() not in water_resnames and resname.upper() not in ion_resnames:
                other_molecules[resname] = count
        info['other_molecules'] = other_molecules

        # Box information (from first frame)
        self.read_frame(0)
        box = self.get_box()
        if box is not None and len(box) == 3:
            # get_box() already returns values in Angstroms
            info['box_x'] = float(box[0][0])
            info['box_y'] = float(box[1][1])
            info['box_z'] = float(box[2][2])
            # Calculate volume (already in Ų)
            info['box_volume'] = float(box[0][0] * box[1][1] * box[2][2])

            # Store full box matrix for triclinic cells (already in Angstroms)
            info['box_matrix'] = [
                [float(box[i][j]) for j in range(3)] for i in range(3)]

        # Protein-specific information
        if self.protein_molecules:
            protein_atoms = self.universe.select_atoms('protein')
            info['n_protein_atoms'] = len(protein_atoms)
            info['n_protein_residues'] = len(protein_atoms.residues)

            # Protein residue composition
            protein_residues = {}
            for residue in protein_atoms.residues:
                resname = residue.resname
                protein_residues[resname] = protein_residues.get(
                    resname, 0) + 1
            info['protein_residue_counts'] = protein_residues

            # Protein sequence (if available)
            try:
                sequence = ''.join(
                    [residue.resname for residue in protein_atoms.residues])
                info['protein_sequence'] = sequence[:100]  # First 100 residues
            except:
                pass

        # Charge information (if available)
        try:
            if hasattr(self.universe.atoms, 'charges'):
                total_charge = np.sum(self.universe.atoms.charges)
                info['total_charge'] = float(total_charge)
        except:
            pass

        # Mass information
        try:
            if hasattr(self.universe.atoms, 'masses'):
                total_mass = np.sum(self.universe.atoms.masses)
                info['total_mass'] = float(total_mass)
        except:
            pass

        # Density calculation (if we have mass and volume)
        if 'total_mass' in info and 'box_volume' in info and info['box_volume'] > 0:
            # Convert: mass in amu, volume in Angstrom^3
            # Density = mass(amu) / volume(A^3) * 1.66054 g/cm^3
            info['density_g_cm3'] = info['total_mass'] / \
                info['box_volume'] * 1.66054

        return info

    def print_system_summary(self, verbose=True):
        """
        Print a formatted summary of the system information.

        Args:
            verbose: If True, print detailed information
        """
        info = self.get_system_info()

        print("\n" + "="*60)
        print("SYSTEM INFORMATION SUMMARY")
        print("="*60)

        print(f"\nTopology File: {info.get('topology_file', 'N/A')}")
        print(f"Total Atoms: {info.get('n_atoms', 0)}")
        print(f"Total Residues: {info.get('n_residues', 0)}")
        print(f"Total Frames: {info.get('n_frames', 0)}")

        if 'dt' in info and info['dt']:
            print(f"Time Step: {info['dt']:.3f} ps")
        if 'total_time' in info and info['total_time']:
            print(f"Total Simulation Time: {info['total_time']:.2f} ps")

        print(f"\n--- Molecular Composition ---")
        print(f"Proteins: {info.get('n_proteins', 0)}")
        print(f"Water Molecules: {info.get('n_water_molecules', 0)}")
        print(f"Ion Molecules: {info.get('n_ion_molecules', 0)}")
        print(f"Other Molecules: {info.get('n_other_molecules', 0)}")

        if 'detected_water_model' in info and info['detected_water_model']:
            print(f"\n--- Water Information ---")
            print(f"Detected Water Model: {info['detected_water_model']}")
            print(f"Water Residue Name: {info.get('water_resname', 'N/A')}")
            print(f"Total Water Atoms: {info.get('n_water_atoms', 0)}")

        if 'ion_counts' in info and info['ion_counts']:
            print(f"\n--- Ion Composition ---")
            for ion, count in info['ion_counts'].items():
                charge_symbol = ""
                if ion in ['Na', 'K']:
                    charge_symbol = "⁺"
                elif ion == 'Cl':
                    charge_symbol = "⁻"
                elif ion in ['Ca', 'Mg']:
                    charge_symbol = "²⁺"
                print(f"{ion}{charge_symbol}: {count}")

        if 'box_x' in info:
            print(f"\n--- Box Dimensions ---")
            print(f"X: {info['box_x']:.3f} Å")
            print(f"Y: {info['box_y']:.3f} Å")
            print(f"Z: {info['box_z']:.3f} Å")
            print(f"Volume: {info['box_volume']:.2f} Å³")

        if 'density_g_cm3' in info:
            print(f"System Density: {info['density_g_cm3']:.4f} g/cm³")

        if 'total_charge' in info:
            print(f"Total System Charge: {info['total_charge']:.3f} e")

        if verbose and 'element_counts' in info:
            print(f"\n--- Element Counts ---")
            for elem, count in sorted(info['element_counts'].items()):
                print(f"{elem}: {count}")

        if verbose and 'protein_residue_counts' in info:
            print(f"\n--- Protein Residue Composition ---")
            total = sum(info['protein_residue_counts'].values())
            for resname, count in sorted(info['protein_residue_counts'].items(),
                                         key=lambda x: x[1], reverse=True)[:10]:
                print(f"{resname}: {count}")
            if len(info['protein_residue_counts']) > 10:
                print(
                    f"... and {len(info['protein_residue_counts']) - 10} more")

        print("\n" + "="*60 + "\n")

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
        return self.ts.triclinic_dimensions

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
        return self.ts.positions

    def iter_frames_stream(self, start: int, stop: int, step: int = 1,
                           show_progress: bool = False,
                           cluster_selection: Optional[str] = None,
                           cluster_cutoff: float = 15.0) -> Iterator[Dict]:
        """
        Memory-efficient streaming iterator for large trajectories.

        This method is optimized for processing very large trajectory files
        (>20GB) by yielding one frame at a time without loading all frames
        into memory. Ideal for SAXS calculations on extensive MD simulations.

        Args:
            start: Starting frame index (inclusive)
            stop: Stopping frame index (exclusive)
            step: Frame stride (default: 1, every frame)
            show_progress: If True, display progress bar (default: False)
            cluster_selection: Selection string for atoms to cluster and reconstruct
                              (default: None, no clustering)
                              Examples: 'resname SDS', 'resname LYS or resname ARG'
            cluster_cutoff: Distance cutoff for DBSCAN clustering in Angstroms (default: 15.0)

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
        trajectory = self.universe.trajectory[start:stop:step]

        if show_progress:
            try:
                from .progress import iter_with_progress
                n_frames = len(range(start, stop, step))
                trajectory = iter_with_progress(
                    trajectory,
                    total=n_frames,
                    desc="Reading frames",
                    unit="frame"
                )
            except ImportError:
                pass  # Progress module not available, continue without progress bar

        # Prepare cluster reconstruction if requested
        cluster_atoms = None
        if cluster_selection is not None:
            try:
                import mdaencore as encore
                cluster_atoms = self.universe.select_atoms(cluster_selection)
                print(
                    f"Cluster reconstruction enabled for {len(cluster_atoms)} atoms using DBSCAN (cutoff={cluster_cutoff} Å)")
            except ImportError:
                print(
                    "Warning: mdaencore not available, skipping cluster reconstruction")
                cluster_atoms = None

        for ts in trajectory:
            # Perform cluster reconstruction if requested
            if cluster_atoms is not None:
                try:
                    # Perform DBSCAN clustering on current frame
                    from sklearn.cluster import DBSCAN
                    from MDAnalysis.lib.mdamath import make_whole

                    # Get positions of selected atoms (residue COM for clustering)
                    residues = cluster_atoms.residues
                    com_positions = np.array(
                        [res.atoms.center_of_mass() for res in residues])

                    # Run DBSCAN clustering
                    clustering = DBSCAN(eps=cluster_cutoff,
                                        min_samples=1, metric='euclidean')
                    labels = clustering.fit_predict(com_positions)

                    # Print number of clusters found
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    print(
                        f"Frame {ts.frame}: Found {n_clusters} clusters, {n_noise} noise points")

                    # Reconstruct each cluster by making molecules whole
                    for cluster_id in set(labels):
                        if cluster_id == -1:  # Skip noise points
                            continue

                        # Get residues in this cluster
                        cluster_residues_idx = np.where(
                            labels == cluster_id)[0]
                        cluster_residues = residues[cluster_residues_idx]

                        # Make each molecule in the cluster whole
                        for res in cluster_residues:
                            try:
                                make_whole(res.atoms, inplace=True)
                            except:
                                # If make_whole fails (e.g., no bonds), try unwrap
                                try:
                                    res.atoms.unwrap(
                                        compound='residues', inplace=True)
                                except:
                                    pass  # Skip if both fail

                except Exception as e:
                    print(
                        f"Warning: Cluster reconstruction failed for frame {ts.frame}: {e}")

            yield {
                'frame': ts.frame,
                'time': ts.time,
                'positions': self.universe.atoms.positions,
                'box': ts.triclinic_dimensions
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

    # Detect water model
    water_model = top.detect_water_model()
    if water_model:
        print(f"\nWater Model: {water_model}")

    # Count individual ion types
    ion_counts = top.count_ions()
    if any(ion_counts.values()):
        print(f"\nIon Composition:")
        for ion_type, count in ion_counts.items():
            if count > 0:
                print(f"  {ion_type}: {count}")

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
    print(
        f"  Box dimensions (Å): {box[0, 0]:.2f} × {box[1, 1]:.2f} × {box[2, 2]:.2f}")
    print(f"  Time: {top.get_time():.2f} ps")
    print(f"  Number of atoms: {len(coords)}")

    # Example: Stream through first 10 frames
    print(f"\nStreaming example (first 10 frames):")
    for frame_data in top.iter_frames_stream(0, min(10, top.n_frames), 1):
        print(f"  Frame {frame_data['frame']}: "
              f"{len(frame_data['positions'])} atoms at {frame_data['time']:.2f} ps")


if __name__ == "__main__":
    main()
