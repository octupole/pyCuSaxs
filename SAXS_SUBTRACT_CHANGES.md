# Changes to saxs-subtract Tool

## Summary of Changes

Modified `pycusaxs/saxs_subtract.py` to improve the user experience and remove auto-scaling functionality.

## Changes Made

### 1. Removed Auto-Scale Functionality
- **Removed**: `--auto-scale` command-line argument
- **Removed**: Auto-calculation of scaling factor based on volumes and densities
- **Simplified**: `get_scaling_factor()` function now only accepts manual user input
- **Why**: Simplifies the workflow and gives users direct control over scaling

### 2. Made Profile ID Selection Interactive
- **Changed**: `--id` argument is now **optional** instead of required
- **Added**: Interactive profile selection when `--id` is not specified
- **Improved**: Lists all available simulated profiles with details before selection
- **Flexible**: Can still provide `--id` directly for scripted/batch processing

### 3. Updated Terminology
- **Changed**: "experimental profile" → "simulated profile" throughout
- **Updated**: Documentation and help messages to reflect simulation context
- **Why**: More accurate terminology for your use case

## New Behavior

### Interactive Mode (No --id specified)
```bash
saxs-subtract --db my_data.db
```

The tool will:
1. Display all available simulated profiles in the database
2. Show details: ID, Water Model, Grid, Supercell, Time, Density
3. Prompt user to select a simulated profile ID
4. Display selected profile details
5. List available reference profiles
6. Prompt user to select a reference profile ID
7. Prompt user to enter scaling factor manually
8. Perform subtraction and save results

### Direct Mode (--id specified)
```bash
saxs-subtract --db my_data.db --id 1
```

The tool will:
1. Load the specified profile ID (validates existence)
2. Display profile details
3. List available reference profiles
4. Prompt user to select a reference profile ID
5. Prompt user to enter scaling factor manually
6. Perform subtraction and save results

## Usage Examples

### Example 1: Fully Interactive
```bash
saxs-subtract --db ./glyco-c11.db
```

**Output:**
```
Loading simulated profiles from: /path/to/glyco-c11.db

Available Simulated Profiles:
ID    Water Model     Grid            Supercell       Time (ps)    Density (g/cm³)
----------------------------------------------------------------------------------------------------
1     TIP3P          128x128x128     256x256x256     100.00       1.0234
2     TIP3P          128x128x128     256x256x256     200.00       1.0245
3     SPCE           128x128x128     256x256x256     150.00       1.0189

Select simulated profile ID to subtract from: 1

Selected Simulated Profile 1:
  Water Model: TIP3P
  Grid: [128, 128, 128]
  ...

Reference database: /path/to/reference.db

Available Reference Profiles:
ID    Water Model     Grid            Supercell       Time (ps)    Density (g/cm³)
----------------------------------------------------------------------------------------------------
1     TIP3P          128x128x128     256x256x256     500.00       0.9970

Select reference profile ID (1-1): 1

Enter scaling factor for reference subtraction: 0.95

Subtracting reference profile 1 (scaled by 0.950000)...
```

### Example 2: Specify Profile ID
```bash
saxs-subtract --db ./glyco-c11.db --id 1
```

Skips the first interactive selection, loads profile 1 directly.

### Example 3: With Custom Output
```bash
saxs-subtract --db ./glyco-c11.db --id 1 -o protein_only.dat
```

### Example 4: With Custom Reference Database
```bash
saxs-subtract --db ./glyco-c11.db --id 1 --ref-db ./my_reference.db
```

### Example 5: Linear Interpolation
```bash
saxs-subtract --db ./glyco-c11.db --id 1 --interp linear
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--db` | Yes | - | Path to database containing simulated profiles |
| `--id` | No | Interactive | Profile ID to process (prompts if not specified) |
| `--ref-db` | No | Package reference DB | Path to reference database |
| `-o, --output` | No | `subtracted_<id>.dat` | Output file path |
| `--interp` | No | `cubic` | Interpolation method (`linear` or `cubic`) |

## Removed Arguments

| Argument | Reason for Removal |
|----------|-------------------|
| `--auto-scale` | Removed auto-calculation; users now always enter scaling manually |

## Benefits of Changes

1. **Better User Experience**:
   - See all available profiles before choosing
   - No need to look up profile IDs separately

2. **More Control**:
   - Manual scaling gives users full control
   - No automatic calculations that might be incorrect

3. **Flexibility**:
   - Interactive mode for exploration
   - Direct ID specification for scripting

4. **Clearer Workflow**:
   - Step-by-step process
   - Clear prompts and feedback

5. **Accurate Terminology**:
   - "Simulated profiles" reflects actual use case
   - No confusion with experimental data

## Technical Details

### Function Changes

**Before:**
```python
def get_scaling_factor(user_profile: dict, ref_profile: dict, auto: bool = False) -> float:
    if auto:
        # Auto-calculation code...
    # Manual input code...
```

**After:**
```python
def get_scaling_factor() -> float:
    # Only manual input code...
```

### Main Function Logic

**Added:**
```python
# Interactive profile selection
user_profiles = list_profiles(user_db, "Available Simulated Profiles")

if args.id is not None:
    user_id = args.id
    user_profile = user_db.get_profile(user_id)
else:
    # Interactive selection
    user_id = get_user_choice(
        f"\nSelect simulated profile ID to subtract from: ",
        min(p['id'] for p in user_profiles),
        max(p['id'] for p in user_profiles)
    )
    user_profile = user_db.get_profile(user_id)
```

## Testing

To test the changes:

```bash
# Test interactive mode
saxs-subtract --db your_database.db

# Test direct mode
saxs-subtract --db your_database.db --id 1

# Test help message
saxs-subtract --help
```

## Backward Compatibility

**Breaking Changes:**
- `--auto-scale` flag removed (will cause error if used)
- `--id` now optional (existing scripts with `--id` still work)

**Migration:**
- Remove `--auto-scale` from any scripts
- Scripts using `--id` will continue to work unchanged

## Files Modified

- `pycusaxs/saxs_subtract.py` - Main tool implementation

---

**Date**: 2025-01-13
**Version**: Updated for pyCuSaxs 0.1.0+
