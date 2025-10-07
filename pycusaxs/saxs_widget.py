#!/usr/bin/env python3
"""PySide6 widget for configuring SAXS CLI parameters."""

from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Dict, Any

from .saxs_defaults import SaxsDefaults

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QPlainTextEdit,
    QLabel,
    QMessageBox,
    QFileDialog,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QPixmap, QClipboard


class RequiredParametersWidget(QWidget):
    """Widget that holds the required parameters for the SAXS calculation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Load settings
        self.settings = QSettings("pyCuSaxs", "SaxsWidget")

        # Topology file with browse button
        topology_row = QWidget()
        topology_layout = QHBoxLayout(topology_row)
        topology_layout.setContentsMargins(0, 0, 0, 0)
        self.topology_edit = QLineEdit()
        self.topology_edit.setPlaceholderText("Path to topology file")
        # Restore previous value
        self.topology_edit.setText(self.settings.value("topology_file", ""))
        topology_browse_btn = QPushButton("Browse...")
        topology_browse_btn.clicked.connect(self._browse_topology)
        topology_layout.addWidget(self.topology_edit)
        topology_layout.addWidget(topology_browse_btn)
        layout.addRow("Topology (-s)", topology_row)

        # Trajectory file with browse button
        trajectory_row = QWidget()
        trajectory_layout = QHBoxLayout(trajectory_row)
        trajectory_layout.setContentsMargins(0, 0, 0, 0)
        self.trajectory_edit = QLineEdit()
        self.trajectory_edit.setPlaceholderText("Path to trajectory file")
        # Restore previous value
        self.trajectory_edit.setText(self.settings.value("trajectory_file", ""))
        trajectory_browse_btn = QPushButton("Browse...")
        trajectory_browse_btn.clicked.connect(self._browse_trajectory)
        trajectory_layout.addWidget(self.trajectory_edit)
        trajectory_layout.addWidget(trajectory_browse_btn)
        layout.addRow("Trajectory (-x)", trajectory_row)

        # Grid size and frame range in one row
        grid_and_frames_row = QWidget()
        grid_and_frames_layout = QHBoxLayout(grid_and_frames_row)
        grid_and_frames_layout.setContentsMargins(0, 0, 0, 0)
        grid_and_frames_layout.setSpacing(12)

        default_grid = int(SaxsDefaults.GRID_SIZE)

        # Grid size (nx, ny, nz)
        grid_and_frames_layout.addWidget(QLabel("nx:"))
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(SaxsDefaults.GRID_SIZE_RANGE_MIN,
                              SaxsDefaults.GRID_SIZE_RANGE_MAX)
        self.nx_spin.setValue(default_grid)
        grid_and_frames_layout.addWidget(self.nx_spin)

        grid_and_frames_layout.addWidget(QLabel("ny:"))
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(SaxsDefaults.GRID_SIZE_RANGE_MIN,
                              SaxsDefaults.GRID_SIZE_RANGE_MAX)
        self.ny_spin.setValue(default_grid)
        grid_and_frames_layout.addWidget(self.ny_spin)

        grid_and_frames_layout.addWidget(QLabel("nz:"))
        self.nz_spin = QSpinBox()
        self.nz_spin.setRange(SaxsDefaults.GRID_SIZE_RANGE_MIN,
                              SaxsDefaults.GRID_SIZE_RANGE_MAX)
        self.nz_spin.setValue(default_grid)
        grid_and_frames_layout.addWidget(self.nz_spin)

        # Frame range
        grid_and_frames_layout.addWidget(QLabel("Initial (-b):"))
        self.begin_spin = QSpinBox()
        self.begin_spin.setRange(
            SaxsDefaults.GRID_SIZE_RANGE_MIN, SaxsDefaults.GRID_SIZE_RANGE_MAX)
        self.begin_spin.setValue(SaxsDefaults.INITIAL_FRAME)
        grid_and_frames_layout.addWidget(self.begin_spin)

        grid_and_frames_layout.addWidget(QLabel("Last (-e):"))
        self.end_spin = QSpinBox()
        self.end_spin.setRange(
            SaxsDefaults.GRID_SIZE_RANGE_MIN, SaxsDefaults.GRID_SIZE_RANGE_MAX)
        self.end_spin.setValue(SaxsDefaults.LAST_FRAME)
        grid_and_frames_layout.addWidget(self.end_spin)

        grid_and_frames_layout.addStretch()
        layout.addRow("Grid size (-g)", grid_and_frames_row)

    def _browse_topology(self) -> None:
        """Open file dialog to select topology file."""
        # Start from previously used directory
        start_dir = self.settings.value("topology_dir", "")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Topology File",
            start_dir,
            "Topology Files (*.tpr *.pdb *.gro);;All Files (*)"
        )
        if file_path:
            self.topology_edit.setText(file_path)
            # Save file path and directory for next time
            self.settings.setValue("topology_file", file_path)
            import os
            self.settings.setValue("topology_dir", os.path.dirname(file_path))

    def _browse_trajectory(self) -> None:
        """Open file dialog to select trajectory file."""
        # Start from previously used directory
        start_dir = self.settings.value("trajectory_dir", "")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Trajectory File",
            start_dir,
            "Trajectory Files (*.xtc *.trr *.dcd);;All Files (*)"
        )
        if file_path:
            self.trajectory_edit.setText(file_path)
            # Save file path and directory for next time
            self.settings.setValue("trajectory_file", file_path)
            import os
            self.settings.setValue("trajectory_dir", os.path.dirname(file_path))

    def parameters(self) -> Dict[str, Any]:
        """Return the required parameters as a dictionary."""
        return OrderedDict(
            [
                ("topology", self.topology_edit.text()),
                ("trajectory", self.trajectory_edit.text()),
                ("grid_size", (self.nx_spin.value(),
                 self.ny_spin.value(), self.nz_spin.value())),
                ("initial_frame", self.begin_spin.value()),
                ("last_frame", self.end_spin.value()),
            ]
        )


class AdvancedParametersWidget(QWidget):
    """Widget that holds the advanced (optional) parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Load settings
        self.settings = QSettings("pyCuSaxs", "SaxsWidget")

        # Output file with browse button
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output file path")
        # Restore previous value, or use default
        saved_output = self.settings.value("output_file", "")
        self.out_edit.setText(saved_output if saved_output else SaxsDefaults.OUTPUT)
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(self.out_edit)
        output_layout.addWidget(output_browse_btn)
        layout.addRow("Output (-o)", output_row)

        self.dt_spin = QSpinBox()
        self.dt_spin.setRange(SaxsDefaults.DT_RANGE_MIN,
                              SaxsDefaults.DT_RANGE_MAX)
        self.dt_spin.setSingleStep(1)
        self.dt_spin.setValue(SaxsDefaults.DT)
        layout.addRow("Frame interval (--dt)", self.dt_spin)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(
            SaxsDefaults.ORDER_RANGE_MIN, SaxsDefaults.ORDER_RANGE_MAX)
        self.order_spin.setValue(SaxsDefaults.ORDER)
        layout.addRow("BSpline order (--order)", self.order_spin)

        self.grid_scaled_spin = QSpinBox()
        self.grid_scaled_spin.setRange(
            SaxsDefaults.GRID_SCALED_RANGE_MIN, SaxsDefaults.GRID_SCALED_RANGE_MAX)
        self.grid_scaled_spin.setValue(SaxsDefaults.GRID_SCALED)
        layout.addRow("Scaled grid size (--gridS)", self.grid_scaled_spin)

        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setDecimals(SaxsDefaults.SCALE_FACTOR_DECIMALS)
        self.scale_factor_spin.setRange(
            SaxsDefaults.SCALE_FACTOR_RANGE_MIN, SaxsDefaults.SCALE_FACTOR_RANGE_MAX)
        self.scale_factor_spin.setSingleStep(SaxsDefaults.SCALE_FACTOR_STEP)
        self.scale_factor_spin.setValue(SaxsDefaults.SCALE_FACTOR)
        layout.addRow("Grid scale factor (--Scale)", self.scale_factor_spin)

        self.bin_size_spin = QDoubleSpinBox()
        self.bin_size_spin.setDecimals(SaxsDefaults.BIN_SIZE_DECIMALS)
        self.bin_size_spin.setRange(
            SaxsDefaults.BIN_SIZE_RANGE_MIN, SaxsDefaults.BIN_SIZE_RANGE_MAX)
        self.bin_size_spin.setSingleStep(SaxsDefaults.BIN_SIZE_STEP)
        self.bin_size_spin.setValue(SaxsDefaults.BIN_SIZE)
        layout.addRow("Histogram bin size (--bin/--Dq)", self.bin_size_spin)

        self.qcut_spin = QDoubleSpinBox()
        self.qcut_spin.setDecimals(SaxsDefaults.QCUT_DECIMALS)
        self.qcut_spin.setRange(
            SaxsDefaults.QCUT_RANGE_MIN, SaxsDefaults.QCUT_RANGE_MAX)
        self.qcut_spin.setSingleStep(SaxsDefaults.QCUT_STEP)
        self.qcut_spin.setValue(SaxsDefaults.QCUT)
        layout.addRow("Reciprocal cutoff (-q)", self.qcut_spin)

        self.help_checkbox = QCheckBox("Show CLI help instead of running")
        layout.addRow("Help flag (-h)", self.help_checkbox)

        # Solvent parameters section with auto-detect button
        solvent_header = QLabel("Solvent Parameters")
        solvent_header_font = solvent_header.font()
        solvent_header_font.setBold(True)
        layout.addRow(solvent_header)

        detect_button_row = QWidget()
        detect_button_layout = QHBoxLayout(detect_button_row)
        detect_button_layout.setContentsMargins(0, 0, 0, 0)
        self.detect_solvent_button = QPushButton("Auto-detect from Topology")
        self.detect_solvent_button.clicked.connect(self._detect_solvent_params)
        self.detect_solvent_button.setToolTip("Automatically detect water model and ion counts from topology file")
        detect_button_layout.addWidget(self.detect_solvent_button)
        detect_button_layout.addStretch()
        layout.addRow("", detect_button_row)

        self.water_model_edit = QLineEdit()
        self.water_model_edit.setPlaceholderText("Water model identifier")
        self.water_model_edit.setText(SaxsDefaults.WATER_MODEL)
        layout.addRow("Water model (--water)", self.water_model_edit)

        self.sodium_spin = QDoubleSpinBox()
        self.sodium_spin.setDecimals(SaxsDefaults.SODIUM_DECIMALS)
        self.sodium_spin.setRange(
            SaxsDefaults.SODIUM_RANGE_MIN, SaxsDefaults.SODIUM_RANGE_MAX)
        self.sodium_spin.setSingleStep(SaxsDefaults.SODIUM_STEP)
        self.sodium_spin.setValue(SaxsDefaults.SODIUM)
        layout.addRow("Sodium (--na)", self.sodium_spin)

        self.chlorine_spin = QDoubleSpinBox()
        self.chlorine_spin.setDecimals(SaxsDefaults.CHLORINE_DECIMALS)
        self.chlorine_spin.setRange(
            SaxsDefaults.CHLORINE_RANGE_MIN, SaxsDefaults.CHLORINE_RANGE_MAX)
        self.chlorine_spin.setSingleStep(SaxsDefaults.CHLORINE_STEP)
        self.chlorine_spin.setValue(SaxsDefaults.CHLORINE)
        layout.addRow("Chlorine (--cl)", self.chlorine_spin)

    def _browse_output(self) -> None:
        """Open file dialog to select output file path."""
        # Start from previously used directory
        start_dir = self.settings.value("output_dir", "")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            start_dir,
            "Data Files (*.dat *.txt);;All Files (*)"
        )
        if file_path:
            self.out_edit.setText(file_path)
            # Save file path and directory for next time
            self.settings.setValue("output_file", file_path)
            import os
            self.settings.setValue("output_dir", os.path.dirname(file_path))

    def _detect_solvent_params(self) -> None:
        """Auto-detect water model and ion counts from topology file."""
        # Get the topology file path from parent window
        parent = self.parent()
        while parent and not hasattr(parent, 'required_widget'):
            parent = parent.parent()

        if not parent:
            QMessageBox.warning(self, "Error", "Cannot access topology file path")
            return

        topology_path = parent.required_widget.topology_edit.text()
        trajectory_path = parent.required_widget.trajectory_edit.text()

        if not topology_path or not trajectory_path:
            QMessageBox.warning(
                self, "Missing Files",
                "Please select both topology and trajectory files first."
            )
            return

        try:
            from .topology import Topology
            QMessageBox.information(
                self, "Analyzing",
                "Analyzing topology file. This may take a moment..."
            )

            # Load topology
            topo = Topology(topology_path, trajectory_path)

            # Detect water model
            water_model = topo.detect_water_model()
            if water_model:
                self.water_model_edit.setText(water_model)

            # Count ions
            ion_counts = topo.count_ions()
            if ion_counts['Na'] > 0:
                self.sodium_spin.setValue(ion_counts['Na'])
            if ion_counts['Cl'] > 0:
                self.chlorine_spin.setValue(ion_counts['Cl'])

            # Show summary
            msg_parts = []
            if water_model:
                msg_parts.append(f"Water model: {water_model}")
            if ion_counts['Na'] > 0 or ion_counts['Cl'] > 0:
                msg_parts.append(f"Na+: {ion_counts['Na']}, Cl-: {ion_counts['Cl']}")
            if ion_counts['K'] > 0 or ion_counts['Ca'] > 0 or ion_counts['Mg'] > 0:
                other_ions = f"Other ions - K+: {ion_counts['K']}, Ca2+: {ion_counts['Ca']}, Mg2+: {ion_counts['Mg']}"
                msg_parts.append(other_ions)

            if msg_parts:
                QMessageBox.information(
                    self, "Detection Complete",
                    "Detected:\n" + "\n".join(msg_parts)
                )
            else:
                QMessageBox.information(
                    self, "Detection Complete",
                    "No water or ions detected in topology."
                )

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to analyze topology:\n{str(e)}"
            )

    def parameters(self) -> Dict[str, Any]:
        """Return advanced parameters as a dictionary."""
        grid_scaled = self.grid_scaled_spin.value()
        scale_factor = self.scale_factor_spin.value()

        # Validate: both grid_scaled and scale_factor cannot be zero
        if grid_scaled == 0 and scale_factor == 0.0:
            raise ValueError(
                "Both grid_scaled and scale_factor are zero. "
                "Either set grid_scaled > 0 or scale_factor > 0."
            )

        return OrderedDict(
            [
                ("out", self.out_edit.text()),
                ("dt", self.dt_spin.value()),
                ("order", self.order_spin.value()),
                ("grid_scaled", grid_scaled),
                ("scale_factor", scale_factor),
                ("bin_size", self.bin_size_spin.value()),
                ("qcut", self.qcut_spin.value()),
                ("help", self.help_checkbox.isChecked()),
                ("water_model", self.water_model_edit.text()),
                ("sodium", self.sodium_spin.value()),
                ("chlorine", self.chlorine_spin.value()),
            ]
        )


class SaxsParametersWindow(QWidget):
    """Main window combining required and advanced parameter widgets."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SAXS Parameters")

        self.required_widget = RequiredParametersWidget()
        self.advanced_widget = AdvancedParametersWidget()

        # CLI command preview (read-only, 1-2 rows)
        self.cli_preview = QPlainTextEdit()
        self.cli_preview.setReadOnly(True)
        self.cli_preview.setMaximumHeight(60)  # ~2 rows
        self.cli_preview.setPlaceholderText(
            "Equivalent CLI command will appear here...")
        # Use monospace font for CLI
        from PySide6.QtGui import QFont
        cli_font = QFont("Monospace")
        cli_font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.cli_preview.setFont(cli_font)

        # Execution output (large window)
        self.output_view = QPlainTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setPlaceholderText(
            "Execution output will appear here...")
        # Use monospace font for terminal-like output
        output_font = QFont("Monospace")
        output_font.setStyleHint(QFont.StyleHint.TypeWriter)
        output_font.setPointSize(10)
        self.output_view.setFont(output_font)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Add logo and title at the top
        import os
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 10)

        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            # Scale logo to 25% of original size (50% of 50%)
            scaled_pixmap = pixmap.scaled(
                pixmap.width() // 8,
                pixmap.height() // 8,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(scaled_pixmap)
            header_layout.addWidget(logo_label)

        # Add title and subtitle in the center
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        # Main title
        main_title = QLabel("pyCuSaxs")
        main_title_font = main_title.font()
        main_title_font.setPointSize(24)
        main_title_font.setBold(True)
        main_title.setFont(main_title_font)
        main_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(main_title)

        # Subtitle
        subtitle = QLabel("GPU accelerated SAXS from atomistic simulations")
        subtitle_font = subtitle.font()
        subtitle_font.setPointSize(18)
        subtitle_font.setItalic(True)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(subtitle)

        # stretch factor 1 to center
        header_layout.addWidget(title_container, 1)

        layout.addWidget(header_container)

        layout.addWidget(QLabel("Required Parameters"))
        layout.addWidget(self.required_widget)

        # Buttons row: Advanced on left, Execute/Quit on right
        buttons_container = QWidget()
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(6)

        # Advanced Parameters on the left
        self.advanced_button = QPushButton("Advanced Parameters…")
        self.advanced_button.clicked.connect(self.show_advanced_dialog)
        self.advanced_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        buttons_layout.addWidget(self.advanced_button)

        # Stretch to push Execute/Quit to the right
        buttons_layout.addStretch()

        # Execute button (green)
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute)
        self.execute_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.execute_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 5px 15px; }")
        buttons_layout.addWidget(self.execute_button)

        # Quit button (red)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        self.quit_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.quit_button.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; padding: 5px 15px; }")
        buttons_layout.addWidget(self.quit_button)

        layout.addWidget(buttons_container)

        # CLI preview section with copy button
        cli_label_container = QWidget()
        cli_label_layout = QHBoxLayout(cli_label_container)
        cli_label_layout.setContentsMargins(0, 0, 0, 0)
        cli_label_layout.addWidget(
            QLabel("Equivalent CLI Command (non-default options only)"))
        cli_label_layout.addStretch()

        self.copy_button = QPushButton("Copy")
        self.copy_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.copy_button.clicked.connect(self._copy_cli_command)
        self.copy_button.setToolTip("Copy CLI command to clipboard")
        cli_label_layout.addWidget(self.copy_button)

        layout.addWidget(cli_label_container)
        layout.addWidget(self.cli_preview)

        layout.addWidget(QLabel("Execution Output"))
        layout.addWidget(self.output_view)

        # Prepare the dialog that holds advanced configuration fields.
        self._advanced_dialog = QDialog(self)
        self._advanced_dialog.setWindowTitle("Advanced Parameters")
        dialog_layout = QVBoxLayout(self._advanced_dialog)
        dialog_layout.addWidget(self.advanced_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self._advanced_dialog.accept)
        dialog_layout.addWidget(button_box)

    def show_advanced_dialog(self) -> None:
        self._advanced_dialog.show()

    def _copy_cli_command(self) -> None:
        """Copy the CLI command to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.cli_preview.toPlainText())
        # Optional: Show brief feedback
        self.copy_button.setText("Copied!")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1500, lambda: self.copy_button.setText("Copy"))

    def build_cli_command(self, required_params: Dict[str, Any], advanced_params: Dict[str, Any]) -> str:
        """Build CLI command string showing only non-default parameters."""
        cmd_parts = ["pycusaxs"]

        # Required parameters (always shown)
        cmd_parts.append(f"-s {required_params['topology']}")
        cmd_parts.append(f"-x {required_params['trajectory']}")

        # Grid size (always shown)
        grid = required_params['grid_size']
        if isinstance(grid, tuple) and len(grid) == 3:
            if grid[0] == grid[1] == grid[2]:
                grid_str = str(grid[0])
            else:
                grid_str = f"{grid[0]},{grid[1]},{grid[2]}"
        else:
            grid_str = str(grid)

        if grid_str != SaxsDefaults.GRID_SIZE:
            cmd_parts.append(f"-g {grid_str}")

        # Frame range (show if not default)
        if required_params['initial_frame'] != SaxsDefaults.INITIAL_FRAME:
            cmd_parts.append(f"-b {required_params['initial_frame']}")
        if required_params['last_frame'] != SaxsDefaults.LAST_FRAME:
            cmd_parts.append(f"-e {required_params['last_frame']}")

        # Advanced parameters (only if non-default)
        if advanced_params.get('out', '') != SaxsDefaults.OUTPUT:
            cmd_parts.append(f"-o {advanced_params['out']}")
        if advanced_params.get('dt', SaxsDefaults.DT) != SaxsDefaults.DT:
            cmd_parts.append(f"--dt {advanced_params['dt']}")
        if advanced_params.get('order', SaxsDefaults.ORDER) != SaxsDefaults.ORDER:
            cmd_parts.append(f"--order {advanced_params['order']}")
        if advanced_params.get('grid_scaled', SaxsDefaults.GRID_SCALED) != SaxsDefaults.GRID_SCALED:
            cmd_parts.append(f"--gridS {advanced_params['grid_scaled']}")
        if advanced_params.get('scale_factor', SaxsDefaults.SCALE_FACTOR) != SaxsDefaults.SCALE_FACTOR:
            cmd_parts.append(f"--Scale {advanced_params['scale_factor']}")
        if advanced_params.get('bin_size', SaxsDefaults.BIN_SIZE) != SaxsDefaults.BIN_SIZE:
            cmd_parts.append(f"--bin {advanced_params['bin_size']}")
        if advanced_params.get('qcut', SaxsDefaults.QCUT) != SaxsDefaults.QCUT:
            cmd_parts.append(f"-q {advanced_params['qcut']}")
        if advanced_params.get('water_model', SaxsDefaults.WATER_MODEL) != SaxsDefaults.WATER_MODEL:
            cmd_parts.append(f"--water {advanced_params['water_model']}")
        if advanced_params.get('sodium', SaxsDefaults.SODIUM) != SaxsDefaults.SODIUM:
            cmd_parts.append(f"--na {advanced_params['sodium']}")
        if advanced_params.get('chlorine', SaxsDefaults.CHLORINE) != SaxsDefaults.CHLORINE:
            cmd_parts.append(f"--cl {advanced_params['chlorine']}")

        return " ".join(cmd_parts)

    def execute(self) -> None:
        try:
            required_params = self.required_widget.parameters()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Input", str(error))
            return
        advanced_params = self.advanced_widget.parameters()

        # Update CLI preview
        cli_command = self.build_cli_command(required_params, advanced_params)
        self.cli_preview.setPlainText(cli_command)

        # Show parameters in output (this will be overridden in main.py)
        lines = ["Required Parameters:"]
        for key, value in required_params.items():
            lines.append(f"  {key}: {value}")

        lines.append("\nAdvanced Parameters:")
        for key, value in advanced_params.items():
            lines.append(f"  {key}: {value}")

        message = "\n".join(lines)
        self.output_view.setPlainText(message)
        print(message)


def main() -> int:
    app = QApplication(sys.argv)
    window = SaxsParametersWindow()
    window.resize(480, 640)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
