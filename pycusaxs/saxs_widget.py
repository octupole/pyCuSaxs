#!/usr/bin/env python3
"""PySide6 widget for configuring SAXS CLI parameters."""

from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Dict, Any

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
)
from PySide6.QtCore import Qt


class RequiredParametersWidget(QWidget):
    """Widget that holds the required parameters for the SAXS calculation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.topology_edit = QLineEdit()
        self.topology_edit.setPlaceholderText("Path to topology file")
        layout.addRow("Topology (-s)", self.topology_edit)

        self.trajectory_edit = QLineEdit()
        self.trajectory_edit.setPlaceholderText("Path to trajectory file")
        layout.addRow("Trajectory (-x)", self.trajectory_edit)

        self.begin_spin = QSpinBox()
        self.begin_spin.setRange(0, 9_999_999)

        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 9_999_999)

        self.grid_size_edit = QLineEdit()
        self.grid_size_edit.setPlaceholderText("nx[,ny,nz]")
        self.grid_size_edit.setText("128")

        multi_row = QWidget()
        multi_layout = QHBoxLayout(multi_row)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.setSpacing(12)

        def add_pair(label_text: str, widget: QWidget) -> None:
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignmentFlag.AlignVCenter |
                               Qt.AlignmentFlag.AlignRight)
            multi_layout.addWidget(label)
            multi_layout.addWidget(widget)

        add_pair("Grid size (-g)", self.grid_size_edit)
        add_pair("Initial frame (-b)", self.begin_spin)
        add_pair("Last frame (-e)", self.end_spin)
        multi_layout.addStretch()

        layout.addRow(multi_row)

    def parameters(self) -> Dict[str, Any]:
        """Return the required parameters as a dictionary."""
        return OrderedDict(
            [
                ("topology", self.topology_edit.text()),
                ("trajectory", self.trajectory_edit.text()),
                ("grid_size", self._parse_grid_size()),
                ("initial_frame", self.begin_spin.value()),
                ("last_frame", self.end_spin.value()),
            ]
        )

    def _parse_grid_size(self) -> tuple[int, int, int]:
        """Parse the grid size field, allowing 1 or 3 integer components."""
        raw_value = self.grid_size_edit.text().strip()
        if not raw_value:
            raise ValueError("Grid size must contain 1 or 3 integers.")

        parts = raw_value.replace(",", " ").split()
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:  # pragma: no cover - GUI input validation
            raise ValueError("Grid size entries must be integers.") from exc

        if len(values) == 1:
            return (values[0], values[0], values[0])
        if len(values) == 3:
            return tuple(values)

        raise ValueError("Grid size must contain either 1 or 3 integers.")


class AdvancedParametersWidget(QWidget):
    """Widget that holds the advanced (optional) parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output file path")
        layout.addRow("Output (-o)", self.out_edit)

        self.dt_spin = QSpinBox()
        self.dt_spin.setRange(1, 1_000_000)
        self.dt_spin.setSingleStep(1)
        self.dt_spin.setValue(1)
        layout.addRow("Frame interval (--dt)", self.dt_spin)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(4)
        layout.addRow("BSpline order (--order)", self.order_spin)

        self.grid_scaled_spin = QSpinBox()
        self.grid_scaled_spin.setRange(1, 10_000)
        layout.addRow("Scaled grid size (--gridS)", self.grid_scaled_spin)

        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setDecimals(3)
        self.scale_factor_spin.setRange(0.0, 1_000.0)
        self.scale_factor_spin.setSingleStep(0.1)
        layout.addRow("Grid scale factor (--Scale)", self.scale_factor_spin)

        self.bin_size_spin = QDoubleSpinBox()
        self.bin_size_spin.setDecimals(3)
        self.bin_size_spin.setRange(0.0, 1_000.0)
        self.bin_size_spin.setSingleStep(0.1)
        layout.addRow("Histogram bin size (--bin/--Dq)", self.bin_size_spin)

        self.qcut_spin = QDoubleSpinBox()
        self.qcut_spin.setDecimals(3)
        self.qcut_spin.setRange(0.0, 1_000.0)
        self.qcut_spin.setSingleStep(0.1)
        layout.addRow("Reciprocal cutoff (-q)", self.qcut_spin)

        self.help_checkbox = QCheckBox("Show CLI help instead of running")
        layout.addRow("Help flag (-h)", self.help_checkbox)

        self.water_model_edit = QLineEdit()
        self.water_model_edit.setPlaceholderText("Water model identifier")
        layout.addRow("Water model (--water)", self.water_model_edit)

        self.sodium_spin = QDoubleSpinBox()
        self.sodium_spin.setDecimals(3)
        self.sodium_spin.setRange(0.0, 1_000.0)
        self.sodium_spin.setSingleStep(0.1)
        layout.addRow("Sodium (--na)", self.sodium_spin)

        self.chlorine_spin = QDoubleSpinBox()
        self.chlorine_spin.setDecimals(3)
        self.chlorine_spin.setRange(0.0, 1_000.0)
        self.chlorine_spin.setSingleStep(0.1)
        layout.addRow("Chlorine (--cl)", self.chlorine_spin)

    def parameters(self) -> Dict[str, Any]:
        """Return advanced parameters as a dictionary."""
        return OrderedDict(
            [
                ("out", self.out_edit.text()),
                ("dt", self.dt_spin.value()),
                ("order", self.order_spin.value()),
                ("grid_scaled", self.grid_scaled_spin.value()),
                ("scale_factor", self.scale_factor_spin.value()),
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
        self.output_view = QPlainTextEdit()
        self.output_view.setReadOnly(True)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Required Parameters"))
        layout.addWidget(self.required_widget)

        self.advanced_button = QPushButton("Advanced Parameters…")
        self.advanced_button.clicked.connect(self.show_advanced_dialog)
        layout.addWidget(self.advanced_button)

        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute)
        layout.addWidget(self.execute_button)

        layout.addWidget(QLabel("Result"))
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

    def execute(self) -> None:
        try:
            required_params = self.required_widget.parameters()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Input", str(error))
            return
        advanced_params = self.advanced_widget.parameters()

        lines = ["Required Parameters:"]
        for key, value in required_params.items():
            lines.append(f"  {key}: {value}")

        lines.append("Advanced Parameters:")
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
