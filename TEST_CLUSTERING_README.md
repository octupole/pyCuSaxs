# Testing Cluster Reconstruction Feature

This document explains how to test the cluster reconstruction functionality that uses mdaencore + DBSCAN clustering to identify and reconstruct molecular clusters.

## What It Does

The cluster reconstruction feature:
1. Uses DBSCAN clustering (from sklearn, part of mdaencore dependencies) to identify spatial clusters of molecules
2. Reconstructs each cluster by making molecules whole (unwrapping across periodic boundaries)
3. Outputs the full system coordinates that would be passed to RunSaxs for SAXS calculation

## Test Scripts

### 1. `test_clustering.py` - Process multiple frames (selection only)
Outputs only the clustered selection (e.g., just SDS molecules).

```bash
/opt/miniforge/envs/pycusaxs/bin/python test_clustering.py \
  -s test-clustering/npt_run.tpr \
  -f test-clustering/traj_npt_0-100ns.xtc \
  --selection "resname SDS" \
  --cutoff 15.0 \
  -n 5 \
  -o output_dir
```

### 2. `test_clustering_compare.py` - Compare before/after (selection only)
Side-by-side comparison of original vs reconstructed (selection only).

```bash
/opt/miniforge/envs/pycusaxs/bin/python test_clustering_compare.py \
  -s test-clustering/npt_run.tpr \
  -f test-clustering/traj_npt_0-100ns.xtc \
  --selection "resname SDS" \
  --cutoff 15.0 \
  --frame 0 \
  -o compare_dir
```

### 3. `test_clustering_full.py` - Process with FULL SYSTEM output ⭐
**This is what you want!** Outputs complete system including water, ions, everything.

```bash
/opt/miniforge/envs/pycusaxs/bin/python test_clustering_full.py \
  -s test-clustering/npt_run.tpr \
  -f test-clustering/traj_npt_0-100ns.xtc \
  --selection "resname SDS" \
  --cutoff 15.0 \
  -n 3 \
  -o full_system_output
```

Output: PDB files with ALL 273,498 atoms (water, ions, SDS, everything)

### 4. `test_clustering_full_compare.py` - Compare FULL SYSTEM ⭐⭐
**Best for visualization!** Side-by-side comparison with complete system.

```bash
/opt/miniforge/envs/pycusaxs/bin/python test_clustering_full_compare.py \
  -s test-clustering/npt_run.tpr \
  -f test-clustering/traj_npt_0-100ns.xtc \
  --selection "resname SDS" \
  --cutoff 15.0 \
  --frame 0 \
  -o full_compare_output
```

Output:
- `frame_00000_ORIGINAL_full.pdb` - Before reconstruction (273,498 atoms)
- `frame_00000_RECONSTRUCTED_full.pdb` - After reconstruction (273,498 atoms)

## Example Output

When running with SDS molecules (cutoff=15.0 Å):

```
Frame 0: Found 31 clusters, 0 noise points
  Wrote full_system_test/frame_00000_full_system.pdb (273498 atoms)

These PDB files contain:
  - Total atoms: 273498
  - Water molecules: 86866
  - Na ions: 300
  - Clustered (resname SDS): 12600 atoms
```

## Visualization

Load the PDB files in VMD or PyMOL:

```bash
# VMD
vmd full_compare_output/frame_00000_ORIGINAL_full.pdb \
    full_compare_output/frame_00000_RECONSTRUCTED_full.pdb

# PyMOL
pymol full_compare_output/frame_00000_ORIGINAL_full.pdb \
      full_compare_output/frame_00000_RECONSTRUCTED_full.pdb
```

**What to look for:**
- In ORIGINAL: Some SDS molecules may be split across periodic box boundaries
- In RECONSTRUCTED: Same molecules are now whole/intact within their clusters
- Water and other molecules remain in original positions (only clustered selection is modified)

## Parameters

- `--selection`: MDAnalysis selection string for clustering
  - Examples: `"resname SDS"`, `"protein"`, `"resname LYS or resname ARG"`
- `--cutoff`: DBSCAN distance cutoff in Ångströms (default: 15.0)
  - Smaller cutoff = more, smaller clusters
  - Larger cutoff = fewer, larger clusters
- `-n, --nframes`: Number of frames to process
- `--frame`: Specific frame number for comparison scripts
- `-o, --output`: Output directory for PDB files

## Technical Details

The implementation is in `pycusaxs/topology.py`:
- Method: `iter_frames_stream()`
- New parameters: `cluster_selection` and `cluster_cutoff`
- Uses DBSCAN on residue centers of mass
- Reconstructs each cluster using `make_whole()` or `unwrap()`
- Prints cluster count per frame

This is the exact coordinate data that gets passed to the C++ RunSaxs function for SAXS calculations.
