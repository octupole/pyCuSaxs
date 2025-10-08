# SAXS Profile Database

pyCuSAXS includes functionality to save and manage SAXS solvent profiles in a SQLite database for reuse across simulations.

## Key Features

The database stores SAXS profiles with the following identification parameters:

- **Water model** (TIP3P, TIP4P, SPC, SPCE)
- **Ion composition** (Na⁺, Cl⁻, K⁺, Ca²⁺, Mg²⁺ counts)
- **Box dimensions** (x, y, z in Angstroms)
- **Supercell scale factor**
- **Simulation time analyzed** (ps)

Each profile is uniquely identified by a hash computed from these parameters, preventing duplicate entries.

## Usage

### Save Profile to Database

Add the `--save-db` flag when running a SAXS calculation:

```bash
# Save to default database (saxs_profiles.db)
pycusaxs -s topology.tpr -x trajectory.xtc -g 128 -b 0 -e 1000 \
    --gridS 320 --save-db

# Save to custom database
pycusaxs -s topology.tpr -x trajectory.xtc -g 128 -b 0 -e 1000 \
    --gridS 320 --save-db my_profiles.db
```

### List Profiles in Database

```bash
# List all profiles
python -m pycusaxs.saxs_db_tool list

# List profiles for specific water model
python -m pycusaxs.saxs_db_tool list --water-model TIP3P

# Use custom database
python -m pycusaxs.saxs_db_tool list --db my_profiles.db
```

Example output:
```
================================================================================
SAXS Profiles in saxs_profiles.db
================================================================================
ID    Water    Ions                 Box (Å)                   Scale    Time (ps)
--------------------------------------------------------------------------------
1     TIP3P    Na:38, Cl:38         100.0x100.0x100.0         2.500    50000.0
2     TIP4P    none                 80.0x80.0x80.0            2.000    25000.0
3     SPCE     K:10, Cl:10          120.0x120.0x120.0         3.000    100000.0
--------------------------------------------------------------------------------
Total profiles: 3
```

### Show Detailed Profile Information

```bash
python -m pycusaxs.saxs_db_tool info 1
```

Example output:
```
============================================================
SAXS Profile Details (ID: 1)
============================================================

--- Solvent Information ---
Water Model: TIP3P
Water Molecules: 33216

Ion Composition:
  Na⁺: 38
  Cl⁻: 38

--- Box Dimensions ---
Size: 100.000 x 100.000 x 100.000 Å
Volume: 1000000.00 Å³

--- Supercell ---
Scale Factor: 2.5000
Supercell Volume: 15625000.00 Å³

--- Simulation Parameters ---
Simulation Time: 50000.00 ps
Frames Analyzed: 5000
Frame Stride: 10

--- SAXS Calculation ---
Grid Size: [128, 128, 128]
B-Spline Order: 4
Bin Size: 0.026
Q Cutoff: 1.5

--- System Properties ---
Total Atoms: 99686
Density: 0.9820 g/cm³

--- Profile Data ---
Data Points: 57
Q Range: 0.026000 - 1.482000 Å⁻¹

--- Metadata ---
Created: 2025-10-08 14:23:45
Profile Hash: a3f7e2c8d9b4f1a2...
Notes: Generated from topology.tpr
```

### Export Profile to CSV

```bash
python -m pycusaxs.saxs_db_tool export 1 tip3p_profile.csv
```

The exported CSV includes metadata in comments:
```csv
# SAXS Solvent Profile
# Water Model: TIP3P
# Water Molecules: 33216
# Ion Counts: {'Na': 38, 'Cl': 38}
# Box Size: 100.000 x 100.000 x 100.000 Å
# Supercell Scale: 2.5000
# Simulation Time: 50000.00 ps
# Frames Analyzed: 5000
# Density: 0.9820 g/cm³
#
# q (1/Å), I(q) (1/Å³)
0.026000,1.234567e-03
0.052000,9.876543e-04
...
```

## Database Schema

The SQLite database contains a single table `saxs_profiles` with these key fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | INTEGER | Primary key |
| `profile_hash` | TEXT | Unique hash of identification parameters |
| `water_model` | TEXT | Water model (TIP3P, TIP4P, SPC, SPCE) |
| `n_water_molecules` | INTEGER | Number of water molecules |
| `ion_counts` | TEXT (JSON) | Ion composition as JSON |
| `box_x`, `box_y`, `box_z` | REAL | Box dimensions (Å) |
| `box_volume` | REAL | Box volume (Å³) |
| `supercell_scale` | REAL | Supercell scale factor |
| `supercell_volume` | REAL | Supercell volume (Å³) |
| `simulation_time_ps` | REAL | Simulation time analyzed |
| `n_frames_analyzed` | INTEGER | Number of frames |
| `grid_size` | TEXT (JSON) | Grid dimensions |
| `profile_data` | TEXT (JSON) | I(q) vs q data |

## Python API

You can also use the database programmatically:

```python
from pycusaxs.saxs_database import SaxsDatabase

# Open database
with SaxsDatabase("saxs_profiles.db") as db:
    # Save a profile
    profile_id = db.save_profile(
        profile_data=[(q1, iq1), (q2, iq2), ...],
        water_model="TIP3P",
        n_water_molecules=33216,
        ion_counts={'Na': 38, 'Cl': 38},
        box_x=100.0, box_y=100.0, box_z=100.0,
        box_volume=1000000.0,
        supercell_scale=2.5,
        supercell_volume=15625000.0,
        simulation_time_ps=50000.0,
        n_frames_analyzed=5000,
        grid_size=(128, 128, 128),
        # ... other parameters
    )

    # Find a matching profile
    profile = db.find_profile(
        water_model="TIP3P",
        ion_counts={'Na': 38, 'Cl': 38},
        box_x=100.0, box_y=100.0, box_z=100.0,
        supercell_scale=2.5
    )

    if profile:
        print(f"Found existing profile: ID {profile['id']}")
        data = profile['profile_data']  # List of (q, I(q)) tuples

    # List all profiles
    all_profiles = db.list_profiles()
    tip3p_profiles = db.list_profiles(water_model="TIP3P")
```

## Use Cases

1. **Building a solvent reference library**: Run 50 ns simulations for different solvent systems (pure water with various models, water + salt combinations) and save all profiles to a shared database.

2. **Reusing profiles**: When running SAXS calculations with the same solvent conditions, retrieve the pre-computed profile from the database instead of recalculating.

3. **Comparing solvent models**: Export profiles for different water models or ion concentrations and compare their SAXS signatures.

4. **Version control**: Commit the SQLite database to git along with your source code to share solvent profiles across the team.
