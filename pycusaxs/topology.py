"""
Initializes the Topology class with the given TPR and XTC files, and provides methods to build a graph representation of the molecular topology, classify the molecules into different types, and read and extract information from the XTC trajectory file.

The Topology class has the following methods:

__init__(tpr_file, xtc_file):
    Initializes the Topology object with the given TPR and XTC files, builds the molecular graph, and classifies the molecules.

__build_graph():
    Builds a graph representation of the molecular topology, where atoms are nodes and bonds are edges.

__classify_molecules():
    Classifies the molecules into proteins, waters, ions, and other types based on the residues present in each molecule.

count_molecules():
    Counts the total number of molecules and the number of molecules in each category (proteins, waters, ions, others).

generate_molecule_dict():
    Generates a dictionary with detailed information about each molecule, including the list of atom indices for each molecule.

get_atom_index():
    Returns a dictionary mapping atom types to lists of their corresponding atom indices.

read_frame(frame_number):
    Reads the specified frame from the XTC trajectory file.

get_box():
    Returns the box dimensions for the current frame.

get_step():
    Returns the step number for the current frame.

get_time():
    Returns the time for the current frame.

get_coordinates():
    Returns the coordinates for the atoms in the current frame.
"""
#!/usr/bin/env python
import argparse
import MDAnalysis as mda
from MDAnalysis.lib.formats.libmdaxdr import XTCFile
import networkx as nx
import sys
import numpy as np
import time
import warnings


class Topology:
    def __init__(self, tpr_file, xtc_file):
        """
        Initializes the MoleculeCounter with the given TPR and XTC files.
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
        self.__build_graph()
        self.__classify_molecules()
        print("Initialized Topology")

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

    def count_molecules(self):
        """
        Counts the total number of molecules and categorizes them.
        """

        num_molecules = len(self.molecules)
        num_proteins = len(self.protein_molecules)
        num_waters = len(self.water_molecules)
        num_ions = len(self.ion_molecules)
        num_others = len(self.other_molecules)

        return num_molecules, num_proteins, num_waters, num_ions, num_others

    def generate_molecule_dict(self):
        """
        Generates a dictionary with detailed information about each molecule.
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

    def get_atom_index(self):
        atom_index = {}
        for atom in self.universe.atoms:
            atom_type = atom.element
            if atom_type not in atom_index:
                atom_index[atom_type] = []
            atoms2 = atom.index
            atom_index[atom_type].append(atoms2)
        return atom_index

    def read_frame(self, frame_number):
        self.ts = self.universe.trajectory[frame_number]

    def get_box(self):
        return self.ts.triclinic_dimensions/10.0

    def get_step(self):
        return self.ts.frame

    def get_time(self):
        return self.ts.time

    def get_coordinates(self):
        return self.universe.atoms.positions/10.0


def main():
    """
    Main function to parse arguments and initiate molecule counting.
    """
    parser = argparse.ArgumentParser(
        description='Read TPR and XTC files and count the number of molecules in the system.')
    parser.add_argument('-s', '--tpr', type=str, help='Path to the TPR file')
    parser.add_argument('-x', '--xtc', type=str, help='Path to the XTC file')
    parser.add_argument('-o', '--output', type=str,
                        help='Output PDB file', default='centered_structure.pdb')

    args = parser.parse_args()

    # Check if both arguments are provided
    if not args.tpr or not args.xtc:
        parser.error("XTC and TPR file file must be provided.")
        sys.exit(1)

    # Initialize MoleculeCounter and count molecules
    top = Topology(args.tpr, args.xtc)

    (num_molecules, num_proteins, num_waters,
     num_ions, num_others) = top.count_molecules()

    # Print the results
    print(f'Total number of molecules: {num_molecules}')
    print(f'Number of protein molecules: {num_proteins}')
    print(f'Number of water molecules: {num_waters}')
    print(f'Number of ion molecules: {num_ions}')
    if num_others:
        print(f'Number of other molecules: {num_others}')

    atom_indesx = top.get_atom_index()
    top.read_frame(10)
    box = top.get_box()
    print(top.frame.box[0])
    # for i in range(0,200,20):
    #     print(i)
    #     top.read_frame(i)
    #     box=top.get_box()
    #     print(box)
    #     print(top.get_step())
    #     print(top.get_time())
    # o=0
    # for frame in top.xtc:
    #     if o !=0 and o%20 == 0:
    #         print(frame.box, frame.time, frame.step)
    #     if o>200:
    #         break
    #     o+=1

    # # Generate and print the molecule dictionary
    # molecule_dict = traj.generate_molecule_dict()
    # for molecule_type, molecules in molecule_dict.items():
    #     print(f"{molecule_type}:")
    #     for key, atoms in molecules.items():
    #         print(f"  {key}: {atoms}")

    # Center the structure on the protein barycenter and apply periodic boundary conditions
    # Write the centered structure to a PDB file
    # traj.write_pdb(1000, args.output)

    exit(1)
    print(f"Centered structure written to {args.output}")


if __name__ == "__main__":
    main()
