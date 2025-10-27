# SAXS Profile Database Usage Guide

## Overview

The pyCuSAXS package includes a SQLite-based database system for storing and managing SAXS profiles with comprehensive metadata. The system maintains two separate databases:

1. **User Database** - Read-write database for your simulation results
2. **Reference Database** - Read-only database containing validated reference solvent profiles

## Database Locations

### User Database (Read-Write)
```
~/.local/share/pycusaxs/user_profiles.db
```
or
```
$XDG_DATA_HOME/pycusaxs/user_profiles.db
```

### Reference Database (Read-Only)
```
<env>/lib/python3.x/site-packages/pycusaxs/data/reference_solvents.db
```

Example for conda environment:
```
/opt/miniforge3/envs/pycusaxs/lib/python3.13/site-packages/pycusaxs/data/reference_solvents.db
```

## Command-Line Tool: `saxs-db`

The `saxs-db` command provides a comprehensive interface for database management.

### Basic Usage

```bash
# By default, operates on user database
saxs-db list

# Use reference database
saxs-db --use-reference list

# Use custom database
saxs-db --db /path/to/custom.db list
```

## Available Commands

### 1. List Profiles

Display all profiles in the database with summary information.

```bash
# List all profiles in user database
saxs-db list

# List all profiles in reference database
saxs-db --use-reference list

# Filter by water model
saxs-db list --water-model TIP3P
saxs-db --use-reference list --water-model SPCE
```

**Output Example:**
```
==================================================================================================================================
SAXS Profiles in ~/.local/share/pycusaxs/user_profiles.db
==================================================================================================================================
ID    Water    Ions            Other Molecules      Box (Å)              Grid            Supercell       Time (ps)  Density (g/cm³)
----------------------------------------------------------------------------------------------------------------------------------
1     TIP3P    Na:150, Cl:150  none                 80.5x80.5x80.5       128x128x128     320x320x320     1000.0     1.0045
2     SPCE     none            POPC:128             90.0x90.0x70.0       64x64x64        160x160x160     500.0      0.9987
3     TIP4P    Na:10, Cl:10    Protein:1            75.2x75.2x75.2       128x128x128     256x256x256     2000.0     1.0123
----------------------------------------------------------------------------------------------------------------------------------
Total profiles: 3
```

### 2. Show Profile Details

Display comprehensive information about a specific profile.

```bash
# Show details for profile ID 1
saxs-db info 1

# Show details from reference database
saxs-db --use-reference info 1
```

**Output Example:**
```
============================================================
SAXS Profile Details (ID: 1)
============================================================

--- Solvent Information ---
Water Model: TIP3P
Water Molecules: 5000

Ion Composition:
  Na⁺: 150
  Cl⁻: 150

--- Box Dimensions ---
Size: 80.532 x 80.532 x 80.532 Å
Volume: 522103.45 Å³

--- Supercell ---
Scale Factor: 2.5000
Supercell Grid: 320 x 320 x 320
Supercell Volume: 8153741.12 Å³

--- Simulation Parameters ---
Simulation Time: 1000.00 ps
Frames Analyzed: 100
Frame Stride: 10

--- SAXS Calculation ---
Grid Size: [128, 128, 128]
B-Spline Order: 4
Bin Size: 0.01
Q Cutoff: 2.0

--- System Properties ---
Total Atoms: 15300
Density: 1.0045 g/cm³

--- Profile Data ---
Data Points: 200
Q Range: 0.010000 - 2.000000 Å⁻¹

--- Metadata ---
Created: 2025-10-22 14:32:15
Profile Hash: a3f5d8c9e2b1f7a4...
Notes: Generated from system.tpr
============================================================
```

### 3. Export Profile Data

Export profile to various formats for analysis or plotting.

#### CSV Format (with metadata)
```bash
# Export to CSV
saxs-db export 1 profile.csv

# Export from reference database
saxs-db --use-reference export 1 tip3p_reference.csv
```

**CSV Format:**
```csv
# SAXS Profile Export
# Profile ID: 1
# Water Model: TIP3P
# Box Size: 80.532 x 80.532 x 80.532 Å
# Grid: [128, 128, 128]
# Supercell Scale: 2.5000
# Created: 2025-10-22 14:32:15
q,I(q)
0.010000,1.234567e+05
0.020000,9.876543e+04
...
```

#### Plot Format (xmgrace/gnuplot compatible)
```bash
# Export to simple x-y format
saxs-db plot 1 profile.dat

# Export from reference database
saxs-db --use-reference plot 1 tip3p_reference.dat
```

**Plot Format:**
```
# SAXS Profile from Database
# Profile ID: 1
# Water Model: TIP3P
# Box Size: 80.532 x 80.532 x 80.532 Å
# Grid: [128, 128, 128]
# Supercell Scale: 2.5000
# Simulation Time: 1000.00 ps
# Density: 1.0045 g/cm³
#
# q (Å⁻¹)    I(q)
0.010000  1.234567e+05
0.020000  9.876543e+04
...
```

### 4. Delete Profiles

Remove profiles from the user database (not allowed for reference database).

```bash
# Delete single profile (with confirmation)
saxs-db delete 1

# Delete multiple profiles
saxs-db delete 1 2 3

# Delete without confirmation prompt
saxs-db delete 1 -y
saxs-db delete 1 2 3 --yes
```

**Safety Features:**
- Deletion from reference database is **blocked**
- Shows profile details before deletion
- Requires confirmation (unless `-y` flag is used)
- Reports number of profiles deleted

**Example:**
```bash
$ saxs-db delete 2

Profile 2:
  Water Model: SPCE
  Grid: [64, 64, 64]
  Supercell Scale: 2.5000
  Simulation Time: 500.00 ps
  Created: 2025-10-22 15:45:30

Delete 1 profile(s)? [y/N]: y

Deleted 1 profile(s).
```

**Attempting to delete from reference database:**
```bash
$ saxs-db --use-reference delete 1
ERROR: Cannot delete profiles from reference database.
The reference database is read-only and contains reference solvent profiles.
```

## Saving Profiles During SAXS Calculation

Profiles can be automatically saved to the database during SAXS calculation.

### Save to User Database

```bash
# Save to default user database
pycusaxs -s system.tpr -x traj.xtc -g 128 -b 0 -e 100 --save-db

# Save to custom database
pycusaxs -s system.tpr -x traj.xtc -g 128 -b 0 -e 100 --save-db /path/to/custom.db
```

### Save to Reference Database (Admin Only)

**Warning:** This requires write permission to the package installation directory.

```bash
# Save to reference database (for building reference library)
pycusaxs -s pure_water.tpr -x water.xtc -g 128 -b 0 -e 1000 --save-reference
```

## Reference Database Contents

The reference database contains validated SAXS profiles for pure solvents under standard conditions:

- **TIP3P water** - Various box sizes and simulation times
- **TIP4P water** - Various box sizes and simulation times
- **SPC water** - Various box sizes and simulation times
- **SPC/E water** - Various box sizes and simulation times
- **Solvated ions** - Standard concentrations (e.g., 150 mM NaCl)

These profiles can be used for:
- **Solvent subtraction** - Remove pure solvent contribution
- **Validation** - Compare your simulation setup
- **Reference data** - Benchmark new calculations

## Working with Both Databases

### Compare User Profile with Reference

```bash
# View user profile
saxs-db info 1

# View reference profile for same water model
saxs-db --use-reference list --water-model TIP3P
saxs-db --use-reference info 5

# Export both for comparison
saxs-db export 1 user_profile.dat
saxs-db --use-reference export 5 ref_profile.dat

# Plot with xmgrace
xmgrace user_profile.dat ref_profile.dat
```

### Find Matching Reference Profile

```bash
# List references by water model
saxs-db --use-reference list --water-model TIP3P

# Find profiles with similar box size and grid
# (manually scan the list output)
```

## Advanced Usage

### Custom Database Path

Useful for project-specific databases:

```bash
# Create project database
PROJECT_DB="./project_saxs_profiles.db"

# Save simulation results
pycusaxs -s sys1.tpr -x traj1.xtc -g 128 -b 0 -e 100 --save-db $PROJECT_DB
pycusaxs -s sys2.tpr -x traj2.xtc -g 128 -b 0 -e 100 --save-db $PROJECT_DB

# List project profiles
saxs-db --db $PROJECT_DB list

# Export for comparison
saxs-db --db $PROJECT_DB export 1 sys1_profile.dat
saxs-db --db $PROJECT_DB export 2 sys2_profile.dat
```

### Filter and Batch Export

```bash
# Export all TIP3P profiles from reference database
saxs-db --use-reference list --water-model TIP3P | grep -oP '^\d+' | while read id; do
    saxs-db --use-reference export $id "reference_tip3p_${id}.csv"
done
```

## Database Schema

The database uses a single table `saxs_profiles` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `water_model` | TEXT | Water model (TIP3P, TIP4P, SPC, SPCE, etc.) |
| `n_water_molecules` | INTEGER | Number of water molecules |
| `ion_counts` | TEXT | JSON dict of ion counts |
| `box_x`, `box_y`, `box_z` | REAL | Box dimensions (Å) |
| `box_volume` | REAL | Box volume (Å³) |
| `supercell_scale` | REAL | Supercell scaling factor |
| `supercell_volume` | REAL | Supercell volume (Å³) |
| `simulation_time_ps` | REAL | Total simulation time (ps) |
| `n_frames_analyzed` | INTEGER | Number of frames analyzed |
| `grid_size` | TEXT | JSON array [nx, ny, nz] |
| `frame_stride` | INTEGER | Frame stride (dt) |
| `bin_size` | REAL | Histogram bin size |
| `qcut` | REAL | Q cutoff (Å⁻¹) |
| `spline_order` | INTEGER | B-spline interpolation order |
| `density_g_cm3` | REAL | System density (g/cm³) |
| `n_atoms` | INTEGER | Total number of atoms |
| `other_molecules` | TEXT | JSON dict of other molecules |
| `profile_data` | TEXT | JSON array of [q, I(q)] pairs |
| `profile_hash` | TEXT | SHA256 hash of profile data |
| `created_timestamp` | TEXT | Creation timestamp (ISO 8601) |
| `notes` | TEXT | Optional notes/description |

## Python API

For programmatic access:

```python
from pycusaxs.saxs_database import SaxsDatabase
from pycusaxs.saxs_defaults import SaxsDefaults

# Open user database
with SaxsDatabase(SaxsDefaults.get_user_database_path()) as db:
    profiles = db.list_profiles(water_model='TIP3P')
    for p in profiles:
        print(f"ID {p['id']}: {p['water_model']}, {p['n_water_molecules']} waters")

# Open reference database
with SaxsDatabase(SaxsDefaults.get_reference_database_path()) as db:
    profiles = db.list_profiles()
    print(f"Reference database contains {len(profiles)} profiles")
```

## Troubleshooting

### Database Not Found

```bash
$ saxs-db --use-reference list
Error: Database not found: /opt/miniforge3/.../reference_solvents.db
```

**Solution:** The reference database is created when you first save a profile using `--save-reference`, or it may not be distributed with your installation yet.

### Permission Denied

```bash
$ saxs-db --use-reference delete 1
ERROR: Cannot delete profiles from reference database.
```

**Solution:** This is intentional. The reference database is read-only. Use the user database for editable profiles.

### Slow Database Operations

For databases with thousands of profiles, consider:
- Use `--water-model` filter when listing
- Export only needed profiles
- Use custom project databases for specific analyses

## Best Practices

1. **Use reference database for validation** - Compare your results with reference solvents
2. **Save important simulations** - Use `--save-db` to preserve results with metadata
3. **Organize by project** - Use custom database paths for different projects
4. **Document your profiles** - The `notes` field helps track what each profile represents
5. **Regular backups** - Backup your user database regularly
6. **Version control metadata** - Store analysis scripts and database queries in version control

## Version History

- **v0.1.0** - Initial database implementation with user/reference separation
  - Added `--use-reference` flag to `saxs-db`
  - Implemented deletion protection for reference database
  - Comprehensive metadata storage
