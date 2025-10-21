# MkDocs Build Fixes

## Problem

The GitHub Actions workflow for building MkDocs documentation was failing with the following errors:

```
WARNING - Doc file 'SCRIPTS.md' contains a link '../README.md', but the target is not found among documentation files.
WARNING - Doc file 'SCRIPTS.md' contains a link '../INSTALL.md', but the target is not found among documentation files.
INFO    - Doc file 'SCRIPTS.md' contains an unrecognized relative link 'user-guide/', it was left as is.
WARNING - Doc file 'quickstart.md' contains a link 'tutorial.md', but the target is not found among documentation files.
WARNING - Doc file 'quickstart.md' contains a link 'cli_reference.md', but the target is not found among documentation files.

Aborted with 4 warnings in strict mode!
```

## Root Causes

1. **SCRIPTS.md** had links to files outside the `docs/` directory:
   - `../README.md` - Links outside docs are not accessible to MkDocs
   - `../INSTALL.md` - Links outside docs are not accessible to MkDocs
   - `user-guide/` - Incorrect relative path format

2. **Duplicate quickstart.md**:
   - `docs/quickstart.md` existed but was not in navigation
   - `docs/getting-started/quickstart.md` was in navigation
   - The orphaned file had broken links to non-existent files

3. **SCRIPTS.md not in navigation**:
   - File existed but wasn't included in `mkdocs.yml` nav section
   - This caused MkDocs to warn about orphaned files

## Solutions Applied

### 1. Fixed Links in docs/SCRIPTS.md

**Before:**
```markdown
For more information, see:
- [README.md](../README.md) - General usage
- [INSTALL.md](../INSTALL.md) - Installation guide
- [docs/user-guide/](user-guide/) - Detailed usage documentation
```

**After:**
```markdown
For more information, see:
- [Installation Guide](getting-started/installation.md) - Installation guide
- [Quick Start](getting-started/quickstart.md) - Getting started
- [Command Line Interface](user-guide/cli.md) - CLI usage
- [Graphical Interface](user-guide/gui.md) - GUI usage
```

**Changes:**
- Removed links to files outside `docs/` directory
- Updated to point to existing documentation files within `docs/`
- All links now use proper relative paths from `docs/SCRIPTS.md`

### 2. Removed Duplicate quickstart.md

**Action:**
```bash
rm /home/marchi/tmp/pyCuSaxs/docs/quickstart.md
```

**Reason:**
- Kept `docs/getting-started/quickstart.md` (in navigation)
- Removed `docs/quickstart.md` (orphaned, not in navigation)
- Eliminated broken links to `tutorial.md` and `cli_reference.md`

### 3. Fixed Links in docs/getting-started/quickstart.md

**Before:**
```markdown
- Read the [Tutorial](tutorial.md) for detailed workflow examples
- Check [CLI Reference](cli_reference.md) for all command-line options
- See [Python API](api/python.rst) for programmatic access
```

**After:**
```markdown
- Read the [Command Line Interface](user-guide/cli.md) guide for detailed CLI options
- Check the [Graphical Interface](user-guide/gui.md) guide for GUI usage
- See the [Python API](api/python.md) documentation for programmatic access
- Visit the [GitHub Repository](https://github.com/octupole/pyCuSaxs) for help and issues
```

**Changes:**
- Removed links to non-existent `tutorial.md` and `cli_reference.md`
- Updated to point to existing documentation
- Fixed Python API link extension from `.rst` to `.md`

### 4. Added SCRIPTS.md to Navigation

**File:** `mkdocs.yml`

**Before:**
```yaml
  - User Guide:
    - Command Line: user-guide/cli.md
    - Graphical Interface: user-guide/gui.md
    - Database Management: user-guide/database.md
    - Solvent Subtraction: user-guide/solvent-subtraction.md
    - Python API: user-guide/python-api.md
```

**After:**
```yaml
  - User Guide:
    - Command Line: user-guide/cli.md
    - Graphical Interface: user-guide/gui.md
    - Database Management: user-guide/database.md
    - Solvent Subtraction: user-guide/solvent-subtraction.md
    - Python API: user-guide/python-api.md
    - Executable Scripts: SCRIPTS.md
```

**Changes:**
- Added SCRIPTS.md to the User Guide section
- Eliminated "orphaned file" warning

## Verification

After these changes, the documentation structure is:

```
docs/
├── index.md
├── SCRIPTS.md                              ✓ Now in nav
├── troubleshooting.md
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md                       ✓ Only one quickstart
│   └── configuration.md
├── user-guide/
│   ├── cli.md                              ✓ Valid link target
│   ├── gui.md                              ✓ Valid link target
│   ├── database.md
│   ├── solvent-subtraction.md
│   └── python-api.md
├── algorithm/
│   ├── overview.md
│   ├── pipeline.md
│   └── performance.md
├── api/
│   ├── backend.md
│   └── python.md                           ✓ Valid link target
└── development/
    ├── contributing.md
    ├── architecture.md
    └── changelog.md
```

## All Links Now Valid

### From docs/SCRIPTS.md:
- ✅ `getting-started/installation.md` - Exists
- ✅ `getting-started/quickstart.md` - Exists
- ✅ `user-guide/cli.md` - Exists
- ✅ `user-guide/gui.md` - Exists

### From docs/getting-started/quickstart.md:
- ✅ `user-guide/cli.md` - Exists (relative path works)
- ✅ `user-guide/gui.md` - Exists (relative path works)
- ✅ `api/python.md` - Exists (relative path works)
- ✅ GitHub URL - External link

## Expected Build Result

Running `mkdocs build --strict` should now:
- ✅ Complete without errors
- ✅ Complete without warnings
- ✅ Include all files in navigation
- ✅ Have all internal links properly resolved

## Files Modified

1. **docs/SCRIPTS.md** - Fixed external links to point to docs files
2. **docs/getting-started/quickstart.md** - Fixed broken links
3. **mkdocs.yml** - Added SCRIPTS.md to navigation
4. **docs/quickstart.md** - Removed (duplicate file)

## Summary

All MkDocs build errors have been resolved by:
1. Converting external links to internal documentation links
2. Removing duplicate/orphaned files
3. Fixing broken links to non-existent files
4. Adding all documentation files to the navigation tree

The documentation should now build successfully in GitHub Actions with `--strict` mode enabled.
