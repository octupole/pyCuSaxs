pyCuSaxs Documentation
======================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/CUDA-11.0+-green.svg
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: CUDA 11.0+

**pyCuSaxs** is a high-performance CUDA-accelerated pipeline for computing Small-Angle X-ray Scattering (SAXS) profiles from molecular dynamics trajectories. It combines a GPU-optimized C++/CUDA backend with Python-based trajectory processing, offering both command-line and graphical user interfaces.

Features
--------

- **GPU-Accelerated Computing**: Leverages NVIDIA CUDA for high-performance SAXS calculations
- **MDAnalysis Integration**: Native support for GROMACS and other MD trajectory formats
- **Dual Interface**: Command-line tool and PySide6-based GUI
- **Production Ready**: Comprehensive input validation, error handling, and exception translation
- **Memory Efficient**: Streaming trajectory processing with double-buffered frame loading
- **Flexible Configuration**: Extensive parameters for grid sizing, histogram binning, and solvent modeling

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   # Create conda environment with dependencies
   conda create -n pycusaxs python=3.11 cmake fmt pybind11 numpy
   conda activate pycusaxs

   # Install Python dependencies
   pip install PySide6 MDAnalysis networkx

   # Build and install pyCuSaxs
   pip install .

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Command-line interface
   pycusaxs -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 100 -o saxs.dat

   # Graphical interface
   pycusaxs  # No arguments launches GUI
   saxs-widget

Python API
^^^^^^^^^^

.. code-block:: python

   from pycusaxs.topology import Topology
   from pycusaxs.core import run_saxs_calculation

   # Load topology
   topo = Topology("system.tpr", "trajectory.xtc")
   print(f"Atoms: {topo.n_atoms}, Frames: {topo.n_frames}")

   # Run calculation
   required = {
       "topology": "system.tpr",
       "trajectory": "trajectory.xtc",
       "grid_size": (128, 128, 128),
       "initial_frame": 0,
       "last_frame": 100
   }
   advanced = {"dt": 10, "order": 4, "bin_size": 0.01}

   results = run_saxs_calculation(required, advanced)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial
   cli_reference
   gui_reference

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/python
   api/cpp
   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   development/setup
   development/architecture
   development/contributing
   development/testing

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   license
   citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
