"""
SQLite database module for storing SAXS solvent profiles.

This module provides functionality to store and retrieve SAXS profiles
for different solvent systems, allowing reuse of calculated profiles.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib


class SaxsDatabase:
    """
    Manages SQLite database for SAXS solvent profiles.

    Key identification parameters:
    - Water model (TIP3P, TIP4P, SPC, SPCE)
    - Ion composition (Na, Cl, K, Ca, Mg counts)
    - Box size (x, y, z in Angstroms)
    - Supercell scale factor
    - Simulation time analyzed (ps)
    """

    def __init__(self, db_path: str = "saxs_profiles.db"):
        """
        Initialize database connection and create tables if needed.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Main profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saxs_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_hash TEXT UNIQUE NOT NULL,

                -- Solvent identification
                water_model TEXT NOT NULL,
                n_water_molecules INTEGER NOT NULL,

                -- Ion composition (stored as JSON for flexibility)
                ion_counts TEXT NOT NULL,

                -- Other molecules: proteins, ligands, etc. (stored as JSON)
                other_molecules TEXT,

                -- Box dimensions
                box_x REAL NOT NULL,
                box_y REAL NOT NULL,
                box_z REAL NOT NULL,
                box_volume REAL NOT NULL,

                -- Supercell information
                supercell_scale REAL NOT NULL,
                supercell_volume REAL NOT NULL,

                -- Simulation parameters
                simulation_time_ps REAL NOT NULL,
                n_frames_analyzed INTEGER NOT NULL,
                frame_stride INTEGER,

                -- SAXS calculation parameters
                grid_size TEXT NOT NULL,
                bin_size REAL,
                qcut REAL,
                spline_order INTEGER,

                -- System density
                density_g_cm3 REAL,

                -- Additional metadata
                n_atoms INTEGER,
                created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,

                -- Profile data (stored as JSON: list of [q, I(q)] pairs)
                profile_data TEXT NOT NULL
            )
        """)

        # Index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_profile_hash
            ON saxs_profiles(profile_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_water_model
            ON saxs_profiles(water_model)
        """)

        self.conn.commit()

    def _compute_profile_hash(self, water_model: str, ion_counts: Dict[str, int],
                              box_x: float, box_y: float, box_z: float,
                              supercell_scale: float, grid_size: Tuple[int, int, int],
                              simulation_time_ps: float) -> str:
        """
        Compute unique hash for a solvent profile based on key parameters.

        Args:
            water_model: Water model name (TIP3P, etc.)
            ion_counts: Dictionary of ion counts
            box_x, box_y, box_z: Box dimensions in Angstroms
            supercell_scale: Supercell scale factor
            grid_size: Tuple of (nx, ny, nz)
            simulation_time_ps: Total simulation time analyzed

        Returns:
            SHA256 hash string
        """
        # Sort ion counts for consistent hashing
        sorted_ions = sorted(ion_counts.items())

        # Create canonical string representation
        key_string = f"{water_model}|"
        key_string += f"{sorted_ions}|"
        key_string += f"{box_x:.3f},{box_y:.3f},{box_z:.3f}|"
        key_string += f"{supercell_scale:.4f}|"
        key_string += f"{grid_size[0]}x{grid_size[1]}x{grid_size[2]}|"
        key_string += f"{simulation_time_ps:.2f}"

        return hashlib.sha256(key_string.encode()).hexdigest()

    def save_profile(self, profile_data: List[Tuple[float, float]],
                    water_model: str,
                    n_water_molecules: int,
                    ion_counts: Dict[str, int],
                    box_x: float, box_y: float, box_z: float,
                    box_volume: float,
                    supercell_scale: float,
                    supercell_volume: float,
                    simulation_time_ps: float,
                    n_frames_analyzed: int,
                    grid_size: Tuple[int, int, int],
                    frame_stride: int = 1,
                    bin_size: float = None,
                    qcut: float = None,
                    order: int = None,
                    density_g_cm3: float = None,
                    n_atoms: int = None,
                    other_molecules: Dict[str, int] = None,
                    notes: str = None) -> int:
        """
        Save SAXS profile to database.

        Args:
            profile_data: List of (q, I(q)) tuples
            water_model: Water model name
            n_water_molecules: Number of water molecules
            ion_counts: Dictionary mapping ion type to count
            box_x, box_y, box_z: Box dimensions (Angstroms)
            box_volume: Box volume (Angstrom^3)
            supercell_scale: Scale factor for supercell
            supercell_volume: Supercell volume (Angstrom^3)
            simulation_time_ps: Total simulation time analyzed (ps)
            n_frames_analyzed: Number of frames analyzed
            grid_size: Tuple of (nx, ny, nz)
            frame_stride: Frame stride used
            bin_size: Q-space bin size
            qcut: Q-space cutoff
            order: B-spline order
            density_g_cm3: System density
            n_atoms: Total atoms
            other_molecules: Dictionary of other molecule residue names and counts (proteins, ligands, etc.)
            notes: Optional notes

        Returns:
            Database row ID of inserted profile
        """
        profile_hash = self._compute_profile_hash(
            water_model, ion_counts, box_x, box_y, box_z, supercell_scale,
            grid_size, simulation_time_ps
        )

        cursor = self.conn.cursor()

        # Convert data to JSON strings
        ion_counts_json = json.dumps(ion_counts)
        other_molecules_json = json.dumps(other_molecules if other_molecules else {})
        grid_size_json = json.dumps(grid_size)
        profile_data_json = json.dumps(profile_data)

        try:
            cursor.execute("""
                INSERT INTO saxs_profiles (
                    profile_hash, water_model, n_water_molecules, ion_counts, other_molecules,
                    box_x, box_y, box_z, box_volume,
                    supercell_scale, supercell_volume,
                    simulation_time_ps, n_frames_analyzed, frame_stride,
                    grid_size, bin_size, qcut, spline_order,
                    density_g_cm3, n_atoms, notes, profile_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile_hash, water_model, n_water_molecules, ion_counts_json, other_molecules_json,
                box_x, box_y, box_z, box_volume,
                supercell_scale, supercell_volume,
                simulation_time_ps, n_frames_analyzed, frame_stride,
                grid_size_json, bin_size, qcut, order,
                density_g_cm3, n_atoms, notes, profile_data_json
            ))

            self.conn.commit()
            return cursor.lastrowid

        except sqlite3.IntegrityError:
            # Profile with this hash already exists
            print(f"Warning: Profile with hash {profile_hash[:16]}... already exists in database")
            cursor.execute("SELECT id FROM saxs_profiles WHERE profile_hash = ?", (profile_hash,))
            return cursor.fetchone()[0]

    def find_profile(self, water_model: str, ion_counts: Dict[str, int],
                    box_x: float, box_y: float, box_z: float,
                    supercell_scale: float, grid_size: Tuple[int, int, int],
                    simulation_time_ps: float,
                    tolerance: float = 0.01) -> Optional[Dict]:
        """
        Find matching profile in database.

        Args:
            water_model: Water model name
            ion_counts: Dictionary of ion counts
            box_x, box_y, box_z: Box dimensions
            supercell_scale: Supercell scale factor
            grid_size: Tuple of (nx, ny, nz)
            simulation_time_ps: Total simulation time
            tolerance: Tolerance for box dimension matching (fraction)

        Returns:
            Dictionary with profile data or None if not found
        """
        profile_hash = self._compute_profile_hash(
            water_model, ion_counts, box_x, box_y, box_z, supercell_scale,
            grid_size, simulation_time_ps
        )

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM saxs_profiles WHERE profile_hash = ?
        """, (profile_hash,))

        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)

        return None

    def list_profiles(self, water_model: Optional[str] = None) -> List[Dict]:
        """
        List all profiles in database, optionally filtered by water model.

        Args:
            water_model: Optional water model filter

        Returns:
            List of profile dictionaries
        """
        cursor = self.conn.cursor()

        if water_model:
            cursor.execute("""
                SELECT * FROM saxs_profiles
                WHERE water_model = ?
                ORDER BY created_timestamp DESC
            """, (water_model,))
        else:
            cursor.execute("""
                SELECT * FROM saxs_profiles
                ORDER BY created_timestamp DESC
            """)

        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert database row to dictionary with parsed JSON fields."""
        result = dict(row)

        # Parse JSON fields
        result['ion_counts'] = json.loads(result['ion_counts'])
        result['other_molecules'] = json.loads(result['other_molecules']) if result.get('other_molecules') else {}
        result['grid_size'] = json.loads(result['grid_size'])
        result['profile_data'] = json.loads(result['profile_data'])

        return result

    def export_profile_csv(self, profile_id: int, output_path: str):
        """
        Export profile data to CSV file.

        Args:
            profile_id: Database ID of profile
            output_path: Path to output CSV file
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM saxs_profiles WHERE id = ?", (profile_id,))
        row = cursor.fetchone()

        if not row:
            raise ValueError(f"Profile with ID {profile_id} not found")

        profile = self._row_to_dict(row)

        with open(output_path, 'w') as f:
            # Write header with metadata
            f.write(f"# SAXS Solvent Profile\n")
            f.write(f"# Water Model: {profile['water_model']}\n")
            f.write(f"# Water Molecules: {profile['n_water_molecules']}\n")
            f.write(f"# Ion Counts: {profile['ion_counts']}\n")
            f.write(f"# Box Size: {profile['box_x']:.3f} x {profile['box_y']:.3f} x {profile['box_z']:.3f} Å\n")
            f.write(f"# Supercell Scale: {profile['supercell_scale']:.4f}\n")
            f.write(f"# Simulation Time: {profile['simulation_time_ps']:.2f} ps\n")
            f.write(f"# Frames Analyzed: {profile['n_frames_analyzed']}\n")
            f.write(f"# Density: {profile['density_g_cm3']:.4f} g/cm³\n")
            f.write(f"#\n")
            f.write(f"# q (1/Å), I(q) (1/Å³)\n")

            # Write data
            for q, iq in profile['profile_data']:
                f.write(f"{q:.6f},{iq:.6e}\n")

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
