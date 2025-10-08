# Database Management

pyCuSAXS includes an SQLite database system for storing and managing SAXS profiles, with special support for solvent subtraction workflows.

## Overview

The database system uses two separate databases:

- **Reference Database**: Read-only, contains pure solvent profiles (shipped with package)
- **User Database**: Read-write, stores your protein+solvent calculations

## Database Locations

| Database | Path | Purpose |
|----------|------|---------|
| Reference | `<package>/data/reference_solvents.db` | Pure solvent reference profiles |
| User | `~/.local/share/pycusaxs/user_profiles.db` | Your SAXS calculations |

## Saving Profiles to Database

### Save to User Database

Use `--save-db` flag to save your calculations:

```bash
pycusaxs -s protein.tpr -x protein.xtc -g 128 -b 0 -e 10000 \
    --gridS 320 --save-db
```

This saves to the default user database: `~/.local/share/pycusaxs/user_profiles.db`

### Custom Database Location

Specify a custom path:

```bash
pycusaxs -s protein.tpr -x protein.xtc -g 128 --save-db /path/to/my_profiles.db
```

### Save to Reference Database (Maintainers Only)

For building the reference solvent library:

```bash
pycusaxs -s tip3p_water.tpr -x tip3p_water.xtc -g 128 -b 0 -e 50000 \
    --gridS 320 --save-reference
```

!!! warning
    The `--save-reference` flag requires write permission to the package directory and should only be used by maintainers building the reference library.

## Stored Information

Each profile stores:

- **Solvent identification**: Water model (TIP3P, TIP4P, SPC, SPCE), ion composition
- **Box dimensions**: x, y, z dimensions and volume
- **Supercell information**: Scale factor and volume
- **Simulation parameters**: Time analyzed, frames, stride
- **SAXS parameters**: Grid size, bin size, q-cutoff, order
- **Profile data**: Full I(q) vs q curve

## Database Management Tool

Use `saxs_db_tool` to manage profiles:

### List Profiles

```bash
# List user profiles
python -m pycusaxs.saxs_db_tool list

# List reference profiles
python -m pycusaxs.saxs_db_tool list --db /path/to/reference_solvents.db

# Filter by water model
python -m pycusaxs.saxs_db_tool list --water-model TIP3P
```

Example output:

```
================================================================================
ID    Water    Ions                 Box (Å)                   Scale    Time (ps)
--------------------------------------------------------------------------------
1     TIP3P    Na:38, Cl:38         100.0x100.0x100.0         2.500    50000.0
2     TIP4P    none                 80.0x80.0x80.0            2.000    25000.0
3     SPCE     K:10, Cl:10          120.0x120.0x120.0         3.000    100000.0
--------------------------------------------------------------------------------
Total profiles: 3
```

### Show Profile Details

```bash
python -m pycusaxs.saxs_db_tool info 1
```

Displays comprehensive information:

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
```

### Export Profile to CSV

```bash
python -m pycusaxs.saxs_db_tool export 1 output.csv
```

Creates a CSV file with metadata header:

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

## Python API

Programmatic access to the database:

```python
from pycusaxs.saxs_database import SaxsDatabase

# Open database
with SaxsDatabase("~/.local/share/pycusaxs/user_profiles.db") as db:
    # List all profiles
    profiles = db.list_profiles()

    # Filter by water model
    tip3p_profiles = db.list_profiles(water_model="TIP3P")

    # Find a specific profile
    profile = db.find_profile(
        water_model="TIP3P",
        ion_counts={'Na': 38, 'Cl': 38},
        box_x=100.0, box_y=100.0, box_z=100.0,
        supercell_scale=2.5
    )

    if profile:
        print(f"Found profile ID: {profile['id']}")
        q_values, i_values = zip(*profile['profile_data'])
```

## Next Steps

- Learn about [Solvent Subtraction](solvent-subtraction.md) workflows
- See the [CLI Guide](cli.md) for all command-line options
- Check [Python API](python-api.md) for programmatic access
