"""
Graphical user interface for pyCuSaxs.

This module provides the main GUI window that integrates the SAXS widget
with the calculation backend.
"""

import sys
from typing import Dict, Any

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QProcess

from .logger import setup_logging, get_logger
from .saxs_widget import SaxsParametersWindow
from .saxs_defaults import SaxsDefaults

logger = get_logger('gui')


class SaxsMainWindow(SaxsParametersWindow):
    """
    Main GUI window for SAXS calculations.

    This window extends SaxsParametersWindow and adds execution logic
    that runs the SAXS calculation when the Execute button is pressed.
    """

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.process = None
        logger.debug("Initialized SAXS main window")

    def execute(self) -> None:
        """
        Execute SAXS calculation when Execute button is pressed.

        This method validates inputs, builds the command-line arguments,
        and starts a QProcess to run the calculation in the background
        with real-time output display.
        """
        # Get parameters from widgets
        try:
            required_params = self.required_widget.parameters()
        except ValueError as error:
            logger.warning(f"Invalid input: {error}")
            QMessageBox.warning(self, "Invalid Input", str(error))
            return

        advanced_params = self.advanced_widget.parameters()

        # Validate grid_scaled and scale_factor
        if (advanced_params.get('grid_scaled', 0) == 0 and
                advanced_params.get('scale_factor', 0.0) == 0.0):
            logger.warning("Both grid_scaled and scale_factor are zero")
            QMessageBox.warning(
                self,
                "Invalid Parameters",
                "Both grid_scaled and scale_factor are zero. "
                "Either set grid_scaled > 0 or scale_factor > 0."
            )
            return

        # Build CLI command for preview
        cli_command = self.build_cli_command(required_params, advanced_params)
        self.cli_preview.setPlainText(cli_command)

        # Clear previous output
        self.output_view.setPlainText("Starting SAXS calculation...\n")
        self.output_view.repaint()

        logger.info("Starting SAXS calculation from GUI")

        # Build command arguments
        args = ["python", "-m", "pycusaxs.main"]

        # Add required arguments
        args.extend(["-s", str(required_params['topology'])])
        args.extend(["-x", str(required_params['trajectory'])])

        # Add grid
        grid = required_params['grid_size']
        if isinstance(grid, tuple) and len(grid) == 3:
            if grid[0] == grid[1] == grid[2]:
                args.extend(["-g", str(grid[0])])
            else:
                args.extend(["-g", f"{grid[0]},{grid[1]},{grid[2]}"])

        # Add frame range
        args.extend(["-b", str(required_params['initial_frame'])])
        args.extend(["-e", str(required_params['last_frame'])])

        # Add advanced parameters if not default
        if advanced_params.get('out'):
            args.extend(["-o", str(advanced_params['out'])])
        if advanced_params.get('dt', SaxsDefaults.DT) != SaxsDefaults.DT:
            args.extend(["--dt", str(advanced_params['dt'])])
        if advanced_params.get('order', SaxsDefaults.ORDER) != SaxsDefaults.ORDER:
            args.extend(["--order", str(advanced_params['order'])])
        if (advanced_params.get('grid_scaled', SaxsDefaults.GRID_SCALED) !=
                SaxsDefaults.GRID_SCALED):
            args.extend(["--gridS", str(advanced_params['grid_scaled'])])
        if (advanced_params.get('scale_factor', SaxsDefaults.SCALE_FACTOR) !=
                SaxsDefaults.SCALE_FACTOR):
            args.extend(["--Scale", str(advanced_params['scale_factor'])])
        if (advanced_params.get('bin_size', SaxsDefaults.BIN_SIZE) !=
                SaxsDefaults.BIN_SIZE):
            args.extend(["--bin", str(advanced_params['bin_size'])])
        if advanced_params.get('qcut', SaxsDefaults.QCUT) != SaxsDefaults.QCUT:
            args.extend(["-q", str(advanced_params['qcut'])])
        if advanced_params.get('water_model'):
            args.extend(["--water", str(advanced_params['water_model'])])
        if advanced_params.get('sodium', SaxsDefaults.SODIUM) != SaxsDefaults.SODIUM:
            args.extend(["--na", str(advanced_params['sodium'])])
        if (advanced_params.get('chlorine', SaxsDefaults.CHLORINE) !=
                SaxsDefaults.CHLORINE):
            args.extend(["--cl", str(advanced_params['chlorine'])])

        # Create and setup QProcess
        self.process = QProcess(self)
        self.process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels
        )

        # Connect signals for real-time output
        def append_output():
            data = (self.process.readAllStandardOutput()
                   .data()
                   .decode('utf-8', errors='replace'))
            self.output_view.appendPlainText(data.rstrip())
            self.output_view.verticalScrollBar().setValue(
                self.output_view.verticalScrollBar().maximum()
            )

        def handle_finished(exit_code, exit_status):
            if exit_code == 0:
                self.output_view.appendPlainText("\n=== Completed Successfully ===")
                logger.info("SAXS calculation completed successfully")
            else:
                self.output_view.appendPlainText(
                    f"\n=== Process exited with code {exit_code} ==="
                )
                logger.error(f"SAXS calculation failed with exit code {exit_code}")

        def handle_error(error):
            error_msg = f"Process error: {error}"
            self.output_view.appendPlainText(error_msg)
            logger.error(error_msg)
            QMessageBox.critical(self, "Execution Error", error_msg)

        self.process.readyReadStandardOutput.connect(append_output)
        self.process.finished.connect(handle_finished)
        self.process.errorOccurred.connect(handle_error)

        # Start the process
        logger.debug(f"Executing command: {' '.join(args)}")
        self.process.start(args[0], args[1:])

        if not self.process.waitForStarted():
            logger.error("Failed to start SAXS process")
            QMessageBox.critical(self, "Error", "Failed to start process")


def run_gui() -> int:
    """
    Launch the GUI application.

    Returns:
        Exit code from the application
    """
    # Setup logging for GUI mode
    setup_logging(verbose=False)
    logger.info("Starting pyCuSaxs GUI")

    app = QApplication([sys.argv[0]])
    window = SaxsMainWindow()
    window.resize(640, 720)
    window.show()

    return app.exec()
