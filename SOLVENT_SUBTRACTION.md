# Solvent Subtraction Workflow

This document describes the complete workflow for SAXS solvent subtraction in pyCuSAXS.

## Overview

The workflow separates protein and solvent contributions using two databases:

1. **Reference Database** (`data/reference_solvents.db`) - Read-only, shipped with package
2. **User Database** (`~/.local/share/pycusaxs/user_profiles.db`) - Read-write, user's calculations

## Architecture

```
┌─────────────────────────────────────────┐
│   Reference Solvent Database            │
│   (data/reference_solvents.db)          │
│   • Read-only                            │
│   • 4-5 pure solvent profiles            │
│   • Shipped with package                 │
│   • Tracked in git                       │
└─────────────────────────────────────────┘
                    │
                    │ Subtract
                    ↓
┌─────────────────────────────────────────┐
│   User Database                          │
│   (~/.local/share/pycusaxs/              │
│    user_profiles.db)                     │
│   • Read-write                           │
│   • User's protein+solvent calculations  │
│   • Not tracked in git                   │
└─────────────────────────────────────────┘
                    │
                    ↓
        ┌───────────────────┐
        │  Protein Profile  │
        │  (pure protein)   │
        └───────────────────┘
```

## Step-by-Step Guide

### Step 1: Build Reference Library (Maintainer/One-time Setup)

Run 50 ns MD simulations for pure solvents:

```bash
# TIP3P water (no protein)
pycusaxs -s tip3p_water.tpr -x tip3p_water.xtc \
    -g 128 -b 0 -e 50000 --gridS 320 \
    --save-reference

# TIP4P water
pycusaxs -s tip4p_water.tpr -x tip4p_water.xtc \
    -g 128 -b 0 -e 50000 --gridS 320 \
    --save-reference

# SPC water
pycusaxs -s spc_water.tpr -x spc_water.xtc \
    -g 128 -b 0 -e 50000 --gridS 320 \
    --save-reference

# SPCE water
pycusaxs -s spce_water.tpr -x spce_water.xtc \
    -g 128 -b 0 -e 50000 --gridS 320 \
    --save-reference
```

This creates `data/reference_solvents.db` with 4 profiles.

### Step 2: Verify Reference Database

```bash
python -m pycusaxs.saxs_db_tool list \
    --db /home/marchi/pyCuSaxs/data/reference_solvents.db
```

Expected output:
```
================================================================================
ID    Water    Ions    Box (Å)              Scale    Time (ps)
--------------------------------------------------------------------------------
1     TIP3P    none    100.0x100.0x100.0    2.500    50000.0
2     TIP4P    none    100.0x100.0x100.0    2.500    50000.0
3     SPC      none    100.0x100.0x100.0    2.500    50000.0
4     SPCE     none    100.0x100.0x100.0    2.500    50000.0
--------------------------------------------------------------------------------
```

### Step 3: Calculate Protein+Solvent SAXS (User)

Run SAXS calculation on your protein+solvent system:

```bash
pycusaxs -s lysozyme_tip3p.tpr -x lysozyme_tip3p.xtc \
    -g 128 -b 0 -e 10000 --gridS 320 \
    -o lysozyme_total.dat \
    --save-db
```

This:
- Calculates SAXS profile for entire system (protein + solvent)
- Saves to `lysozyme_total.dat`
- Stores in user database (`~/.local/share/pycusaxs/user_profiles.db`)

### Step 4: Check User Database

```bash
python -m pycusaxs.saxs_db_tool list
```

Expected output:
```
ID    Water    Ions    Box (Å)              Scale    Time (ps)
1     TIP3P    Na:38   100.0x100.0x100.0    2.500    10000.0
```

### Step 5: Subtract Solvent

Extract pure protein contribution:

```bash
python -m pycusaxs.saxs_subtract \
    --protein-id 1 \
    --solvent-id 1 \
    -o lysozyme_only.dat
```

This computes: **I_protein(q) = I_total(q) - I_solvent(q)**

Output: `lysozyme_only.dat` contains pure protein scattering

## Alternative: Subtract Using .dat Files

If you don't want to use databases:

```bash
python -m pycusaxs.saxs_subtract \
    --protein-file lysozyme_total.dat \
    --solvent-file tip3p_reference.dat \
    -o lysozyme_only.dat
```

## Command Reference

### Save to Reference Database
```bash
pycusaxs -s topology.tpr -x trajectory.xtc -g 128 --save-reference
```
**Use only for building reference library!**

### Save to User Database (Default)
```bash
pycusaxs -s topology.tpr -x trajectory.xtc -g 128 --save-db
```

### List Reference Profiles
```bash
python -m pycusaxs.saxs_db_tool list \
    --db /path/to/pyCuSaxs/data/reference_solvents.db
```

### List User Profiles
```bash
python -m pycusaxs.saxs_db_tool list
```

### Subtract (Database Mode)
```bash
python -m pycusaxs.saxs_subtract \
    --protein-id <user_db_id> \
    --solvent-id <reference_db_id> \
    -o output.dat
```

### Subtract (File Mode)
```bash
python -m pycusaxs.saxs_subtract \
    --protein-file total.dat \
    --solvent-file solvent.dat \
    -o protein.dat
```

## Important Considerations

### 1. Matching Conditions

For accurate subtraction, ensure:
- **Same water model** (TIP3P, TIP4P, etc.)
- **Similar box size** (interpolation handles small differences)
- **Same supercell scale**

### 2. Ion Handling

If your protein system has ions (e.g., Na⁺, Cl⁻):
- Reference database should be pure water (no ions)
- Subtraction will leave protein + ion contributions
- For most proteins, ion contribution is negligible

### 3. Database Locations

| Database | Path | Purpose |
|----------|------|---------|
| Reference | `<pkg>/data/reference_solvents.db` | Pure solvents (maintainer) |
| User | `~/.local/share/pycusaxs/user_profiles.db` | Your calculations |

### 4. Data Files

Both databases **and** .dat files are created:
- Database: For easy reuse and subtraction
- .dat files: For traditional analysis and plotting

## Troubleshooting

**Q: I can't write to reference database**
```bash
# This is by design! Use --save-db for normal work:
pycusaxs -s protein.tpr -x protein.xtc -g 128 --save-db
```

**Q: Where is my user database?**
```bash
ls ~/.local/share/pycusaxs/user_profiles.db
```

**Q: Can I use different box sizes?**
Yes! The subtraction tool interpolates profiles, so small box size differences are handled automatically.

**Q: What if I need a different solvent not in the reference?**
Contact the maintainer to add it to the reference database, or use file-based subtraction with your own reference.

## Complete Example

```bash
# === Maintainer: Build reference (once) ===
pycusaxs -s tip3p.tpr -x tip3p.xtc -g 128 -b 0 -e 50000 \
    --gridS 320 --save-reference

# === User: Calculate protein system ===
pycusaxs -s lysozyme.tpr -x lysozyme.xtc -g 128 -b 0 -e 10000 \
    --gridS 320 --save-db

# === User: List profiles ===
python -m pycusaxs.saxs_db_tool list  # Check user DB, see ID=1
python -m pycusaxs.saxs_db_tool list \
    --db data/reference_solvents.db   # Check reference, see ID=1

# === User: Subtract ===
python -m pycusaxs.saxs_subtract \
    --protein-id 1 --solvent-id 1 -o lysozyme_protein.dat

# === Result ===
# lysozyme_protein.dat contains pure protein scattering!
```
