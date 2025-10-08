# SAXS Profile Databases

This directory contains the **reference solvent database** shipped with pyCuSAXS.

## Database Structure

### 1. Reference Solvent Database (Read-Only)
**Location**: `data/reference_solvents.db`

- **Purpose**: Reference library of pure solvent SAXS profiles
- **Content**: 4-5 carefully validated solvent systems
- **Usage**: Shipped with package, used for subtraction from protein+solvent profiles
- **Version Control**: Tracked in git
- **User Access**: Read-only

**Reference profiles to include:**
- TIP3P pure water
- TIP4P pure water
- SPC pure water
- SPCE pure water
- TIP3P + 0.15M NaCl (optional)

### 2. User Database (Read-Write)
**Location**: `~/.local/share/pycusaxs/user_profiles.db`

- **Purpose**: User's protein+solvent SAXS calculations
- **Content**: All user-generated SAXS profiles
- **Usage**: Automatically created in user's home directory
- **Version Control**: NOT tracked in git (user-specific data)
- **User Access**: Read-write

## Usage

### Building the Reference Library (Maintainer Only)

Run 50 ns simulations for each reference solvent:

```bash
# TIP3P water
pycusaxs -s tip3p.tpr -x tip3p.xtc -g 128 -b 0 -e 50000 \
    --gridS 320 --save-reference

# TIP4P water
pycusaxs -s tip4p.tpr -x tip4p.xtc -g 128 -b 0 -e 50000 \
    --gridS 320 --save-reference

# SPC water
pycusaxs -s spc.tpr -x spc.xtc -g 128 -b 0 -e 50000 \
    --gridS 320 --save-reference

# SPCE water
pycusaxs -s spce.tpr -x spce.xtc -g 128 -b 0 -e 50000 \
    --gridS 320 --save-reference
```

### User Workflow

1. **Run protein+solvent calculation** (saves to user database):
```bash
pycusaxs -s protein.tpr -x protein.xtc -g 128 -b 0 -e 10000 \
    --gridS 320 --save-db
```

2. **List available reference solvents**:
```bash
python -m pycusaxs.saxs_db_tool list --db data/reference_solvents.db
```

3. **List your protein profiles**:
```bash
python -m pycusaxs.saxs_db_tool list
```

4. **Subtract solvent from protein+solvent**:
```bash
# Using database IDs
python -m pycusaxs.saxs_subtract --protein-id 5 --solvent-id 1 -o protein_only.dat

# Or using .dat files directly
python -m pycusaxs.saxs_subtract \
    --protein-file protein_solvent.dat \
    --solvent-file reference_tip3p.dat \
    -o protein_only.dat
```

## Database Locations

| Database | Location | Access | Purpose |
|----------|----------|--------|---------|
| Reference | `<pkg>/data/reference_solvents.db` | Read-only | Pure solvent profiles (shipped) |
| User | `~/.local/share/pycusaxs/user_profiles.db` | Read-write | User's calculations |

## Example Workflow

```bash
# 1. Calculate protein+solvent SAXS (auto-saves to user DB)
pycusaxs -s lysozyme.tpr -x lysozyme.xtc -g 128 --gridS 320 --save-db

# 2. Check what was saved
python -m pycusaxs.saxs_db_tool list
# Output: ID 1, TIP3P, 33216 waters, lysozyme system

# 3. Subtract reference TIP3P solvent (ID 1 in reference DB)
python -m pycusaxs.saxs_subtract --protein-id 1 --solvent-id 1 -o lysozyme_only.dat

# 4. Result: lysozyme_only.dat contains protein scattering only
```

## Important Notes

1. **Reference database is read-only**: Users cannot modify `reference_solvents.db`
2. **Match simulation conditions**: For accurate subtraction, protein+solvent should use the same:
   - Water model
   - Box size (or similar)
   - Supercell scale
3. **Box size tolerance**: Subtraction tool interpolates, so small box size differences are acceptable
4. **.dat files always saved**: In addition to database storage, SAXS profiles are always written to .dat files
