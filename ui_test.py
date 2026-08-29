"""PySide6와 PyVistaQt를 이용한 Surface 경계 선택 GUI 테스트입니다."""

import math
import sys
import json
from pathlib import Path

import gmsh
import numpy as np
import pyvista as pv
from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtGui import QAction, QDoubleValidator, QIcon
from PySide6.QtWidgets import (
	QApplication,
	QComboBox,
	QCheckBox,
	QDialog,
	QDialogButtonBox,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QListWidget,
	QMainWindow,
	QMessageBox,
	QProgressDialog,
	QPushButton,
	QRadioButton,
	QScrollArea,
	QSizePolicy,
	QSpinBox,
	QSplitter,
	QToolBar,
	QVBoxLayout,
	QWidget,
)
from pyvistaqt import QtInteractor

from pyvista_select_test import (
	actor_key,
	get_external_surface_ids,
	get_gmsh_nodes,
	surface_to_polydata,
)
from fem_solver import (
	run_electrical_fem,
	run_real_battery_gaussian_fem,
	scale_real_battery_result_to_current,
	scale_fem_result_to_current,
)


# 시작 시 빈 Viewer를 보여주려면 False로 둡니다.
# 나중에 grid.step을 자동으로 열고 싶을 때만 True로 바꾸면 됩니다.
AUTO_LOAD_DEFAULT_STEP = False
DEFAULT_RESISTIVITY = 2.10e-7
DEFAULT_REFERENCE_TEMPERATURE = 20.0
MESH_PREVIEW_WARNING_ELEMENTS = 1000000
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "grid_resistance_icon.png"


class NoWheelSpinBox(QSpinBox):
	"""마우스 휠은 값을 바꾸지 않고 상위 스크롤 영역으로 전달합니다."""

	def wheelEvent(self, event):
		event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
	"""마우스 휠은 값을 바꾸지 않고 상위 스크롤 영역으로 전달합니다."""

	def wheelEvent(self, event):
		event.ignore()


class MaterialEditDialog(QDialog):
	"""Material 하나의 이름과 두 reference 값을 입력받는 창입니다."""

	def __init__(self, parent=None, material=None):
		super().__init__(parent)
		self.setWindowTitle("Edit Material" if material else "Add Material")
		layout = QFormLayout(self)

		self.name_edit = QLineEdit(material["name"] if material else "")
		self.resistivity_edit = QLineEdit(
			f"{material['resistivity']:.2e}" if material else f"{DEFAULT_RESISTIVITY:.2e}"
		)
		self.temperature_edit = QLineEdit(
			f"{material['reference_temperature']:.1f}"
			if material
			else f"{DEFAULT_REFERENCE_TEMPERATURE:.1f}"
		)
		self.resistivity_edit.setValidator(QDoubleValidator(0.0, 1.0e100, 16, self))
		self.temperature_edit.setValidator(QDoubleValidator(-1.0e100, 1.0e100, 8, self))
		layout.addRow("Material Name", self.name_edit)
		layout.addRow("Resistivity (Ohm m)", self.resistivity_edit)
		layout.addRow("Reference Temperature (C)", self.temperature_edit)

		buttons = QDialogButtonBox(
			QDialogButtonBox.StandardButton.Save
			| QDialogButtonBox.StandardButton.Cancel
		)
		buttons.accepted.connect(self.accept)
		buttons.rejected.connect(self.reject)
		layout.addRow(buttons)

	def get_material(self):
		"""입력된 내용을 JSON에 저장할 Material record로 반환합니다."""
		return {
			"name": self.name_edit.text().strip(),
			"resistivity": float(self.resistivity_edit.text()),
			"reference_temperature": float(self.temperature_edit.text()),
		}


class MaterialLibraryDialog(QDialog):
	"""등록된 Material 목록과 관리 버튼을 보여주는 창입니다."""

	def __init__(self, parent, materials):
		super().__init__(parent)
		self.setWindowTitle("Material Library")
		self.resize(380, 300)
		self.materials = materials
		layout = QVBoxLayout(self)
		self.list_widget = QListWidget()
		layout.addWidget(self.list_widget)

		button_layout = QHBoxLayout()
		self.add_button = QPushButton("Add")
		self.edit_button = QPushButton("Edit")
		self.delete_button = QPushButton("Delete")
		button_layout.addWidget(self.add_button)
		button_layout.addWidget(self.edit_button)
		button_layout.addWidget(self.delete_button)
		layout.addLayout(button_layout)
		close_button = QPushButton("Close")
		close_button.clicked.connect(self.accept)
		layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

	def refresh(self, selected_name=None):
		"""Material 목록을 다시 그리고 가능한 경우 기존 선택을 유지합니다."""
		self.list_widget.clear()
		self.list_widget.addItems([material["name"] for material in self.materials])
		if selected_name:
			matches = self.list_widget.findItems(selected_name, Qt.MatchFlag.MatchExactly)
			if matches:
				self.list_widget.setCurrentItem(matches[0])
		if self.list_widget.currentRow() < 0 and self.materials:
			self.list_widget.setCurrentRow(0)


class MeshPreviewDialog(QDialog):
	"""현재 Gmsh volume mesh를 읽기 전용으로 보여주는 별도 창입니다."""

	def __init__(self, parent, mesh_grid, analysis_config, node_count, tetrahedron_count):
		super().__init__(parent)
		self.setWindowTitle("3D Mesh Preview")
		self.resize(900, 650)
		self.mesh_grid = mesh_grid
		self.analysis_config = analysis_config
		self.viewer = QtInteractor(self)
		self.viewer.set_background("white")
		layout = QVBoxLayout(self)
		self.info_label = QLabel()
		layout.addWidget(self.info_label)

		mode_layout = QHBoxLayout()
		self.surface_radio = QRadioButton("Surface Mesh")
		self.volume_radio = QRadioButton("Volume Mesh")
		self.surface_radio.setChecked(True)
		self.surface_radio.toggled.connect(self._update_mesh_display)
		mode_layout.addWidget(self.surface_radio)
		mode_layout.addWidget(self.volume_radio)
		mode_layout.addStretch(1)
		layout.addLayout(mode_layout)
		layout.addWidget(self.viewer, stretch=1)

		mesh_info = analysis_config["mesh"]
		self.info_label.setText(
			f"Nodes: {node_count:,} | Tetrahedra: {tetrahedron_count:,} | "
			f"Target Mesh Size: {mesh_info['target_mesh_size_mm']:.6g} mm"
		)
		self._update_mesh_display()

	def _update_mesh_display(self):
		"""Surface 또는 전체 Volume 중 선택한 표시 모드를 갱신합니다."""
		self.viewer.clear()
		if self.surface_radio.isChecked():
			preview_data = self.mesh_grid.extract_surface()
			self.viewer.add_mesh(
				preview_data,
				color="lightgray",
				show_edges=True,
				edge_color="dimgray",
				line_width=1.0,
			)
		else:
			self.viewer.add_mesh(
				self.mesh_grid,
				color="lightgray",
				show_edges=True,
				edge_color="dimgray",
				line_width=0.5,
				opacity=0.65,
			)
		self.viewer.reset_camera()
		self.viewer.render()

	def closeEvent(self, event):
		"""Preview만 닫고 메인 창의 Gmsh 모델에는 영향을 주지 않습니다."""
		self.viewer.close()
		event.accept()


class FemResultsDialog(QDialog):
	"""FEM 결과를 읽기 전용으로 시각화하는 별도 Viewer입니다."""

	def __init__(self, parent, result, boundary_mode=2, model_dimensions=None, real_result=None):
		super().__init__(parent)
		self.setWindowTitle("FEM Results")
		self.resize(1000, 720)
		self.result = result
		self.grid_characterization_result = result
		self.boundary_mode = boundary_mode
		self.model_dimensions = model_dimensions or {}
		self.grid_mode = boundary_mode == 1
		self.real_result = real_result
		self.grid_real_battery_result = real_result
		self.viewer = QtInteractor(self)
		self.viewer.set_background("white")
		self.grid = self._build_result_grid()

		layout = QVBoxLayout(self)
		control_layout = QHBoxLayout()
		control_layout.addWidget(QLabel("Result"))
		self.result_combo = QComboBox()
		if self.grid_mode:
			self.result_combo.addItems(
				[
					"Voltage Drop",
					"Resistance Map",
					"Voltage Drop in Real Battery Model",
					"Current Density in Real Battery Model",
					"Joule Loss Density in Real Battery Model",
				]
			)
			self.result_combo.setCurrentIndex(1)
		else:
			self.result_combo.addItems(
				["Voltage Potential", "Current Density", "Joule Loss Density"]
			)
		self.result_combo.currentIndexChanged.connect(self._update_display)
		control_layout.addWidget(self.result_combo)
		self.view_combo = QComboBox()
		self.view_combo.addItems(["Mid-Plane", "3D Surface", "3D Volume"])
		self.view_combo.setCurrentIndex(0 if self.grid_mode else 1)
		self.view_combo.currentIndexChanged.connect(self._update_display)
		self.view_label = QLabel("View")
		self.view_label.setVisible(self.grid_mode)
		self.view_combo.setVisible(self.grid_mode)
		control_layout.addWidget(self.view_label)
		control_layout.addWidget(self.view_combo)
		self.voltage_mode_combo = QComboBox()
		self.voltage_mode_combo.addItems(["Voltage Drop", "Absolute Potential"])
		self.voltage_mode_combo.currentIndexChanged.connect(self._update_display)
		control_layout.addWidget(self.voltage_mode_combo)
		self.current_density_controls = []
		self.current_density_range_combo = QComboBox()
		range_items = [
			"Full Range", "90th Percentile", "80th Percentile",
			"70th Percentile", "60th Percentile", "50th Percentile",
		]
		if self.grid_mode:
			range_items.append("Custom")
		self.current_density_range_combo.addItems(range_items)
		self.current_density_range_combo.setCurrentIndex(1)
		self.current_density_range_combo.currentIndexChanged.connect(self._update_display)
		self.current_density_scale_combo = QComboBox()
		self.current_density_scale_combo.addItems(["Linear", "Logarithmic"])
		self.current_density_scale_combo.currentIndexChanged.connect(self._update_display)
		self.current_density_quantity_combo = QComboBox()
		self.current_density_quantity_combo.addItems(["Absolute J", "Normalized J"])
		self.current_density_quantity_combo.currentIndexChanged.connect(self._update_display)
		for label_text, widget in (
			("Range", self.current_density_range_combo),
			("Scale", self.current_density_scale_combo),
			("Quantity", self.current_density_quantity_combo),
		):
			control_label = QLabel(label_text)
			control_layout.addWidget(control_label)
			control_layout.addWidget(widget)
			self.current_density_controls.extend([control_label, widget])
		self.current_density_min_spin = QDoubleSpinBox()
		self.current_density_min_spin.setRange(0.0, 1.0e100)
		self.current_density_min_spin.setDecimals(8)
		self.current_density_min_spin.setValue(0.0)
		self.current_density_min_spin.valueChanged.connect(self._update_display)
		self.current_density_max_spin = QDoubleSpinBox()
		self.current_density_max_spin.setRange(0.0, 1.0e100)
		self.current_density_max_spin.setDecimals(8)
		self.current_density_max_spin.setValue(1.0)
		self.current_density_min_spin.setVisible(False)
		self.current_density_max_spin.setVisible(False)
		self.current_density_min_label = QLabel("Min J")
		control_layout.addWidget(self.current_density_min_label)
		control_layout.addWidget(self.current_density_min_spin)
		self.current_density_max_label = QLabel("Max J")
		control_layout.addWidget(self.current_density_max_label)
		control_layout.addWidget(self.current_density_max_spin)
		self.hotspot_check = QCheckBox("Show Hotspot")
		self.hotspot_check.setChecked(True)
		self.hotspot_check.toggled.connect(self._update_display)
		control_layout.addWidget(self.hotspot_check)
		control_layout.addStretch(1)
		layout.addLayout(control_layout)
		layout.addWidget(self.viewer, stretch=1)

		self.summary_label = QLabel()
		self.summary_label.setWordWrap(True)
		layout.addWidget(self.summary_label)
		self._update_summary()
		self._update_display()

	def _build_result_grid(self):
		"""solver 결과 배열을 표시 전용 PyVista UnstructuredGrid로 만듭니다."""
		points_mm = np.asarray(self.result["mesh_points_m"], dtype=float) * 1000.0
		tetrahedra = np.asarray(self.result["tetrahedra_connectivity"], dtype=np.int64)
		cells = np.column_stack(
			[np.full(len(tetrahedra), 4, dtype=np.int64), tetrahedra]
		).ravel()
		cell_types = np.full(len(tetrahedra), pv.CellType.TETRA, dtype=np.uint8)
		grid = pv.UnstructuredGrid(cells, cell_types, points_mm)
		if self.grid_mode and self.real_result is not None:
			result_data = self.real_result
		else:
			result_data = self.result
		# Option 2의 전압 강하는 기준 전위에서 각 위치의 절대 전위를 뺀 값입니다.
		# 따라서 Input(reference potential)에서는 0이고 낮은 전위로 갈수록 증가합니다.
		grid.point_data["Voltage Drop (mV)"] = np.maximum(
			self.result["reference_potential_v"] - self.result["absolute_potential_v"],
			0.0,
		) * 1000.0
		grid.point_data["Real Battery Voltage Drop (mV)"] = result_data["voltage_drop_field_v"] * 1000.0
		grid.point_data["Absolute Potential (V)"] = self.result["absolute_potential_v"]
		grid.cell_data["Current Density (A/mm^2)"] = result_data["actual_current_density_magnitude_a_mm2"]
		grid.cell_data["Real Battery Current Density (A/mm^2)"] = result_data["actual_current_density_magnitude_a_mm2"]
		grid.cell_data["Joule Loss Density (W/m^3)"] = result_data["actual_joule_loss_density"]
		grid.cell_data["Real Battery Joule Loss Density (W/m^3)"] = result_data["actual_joule_loss_density"]
		for name in (
			"Voltage Drop (mV)",
			"Absolute Potential (V)",
			"Current Density (A/mm^2)",
			"Joule Loss Density (W/m^3)",
		):
			if not np.all(np.isfinite(grid[name])):
				raise ValueError(f"결과 scalar {name}에 NaN 또는 infinite 값이 있습니다.")
		return grid

	def _update_summary(self):
		"""저항, 전류, 최대장, hotspot 정보를 간단히 표시합니다."""
		hotspot = self.result["actual_hotspot"]
		centroid = hotspot["centroid_mm"]
		self.summary_label.setText(
			f"Analysis Current: {self.result['analysis_current_a']:.6g} A | "
			f"Reference Potential: {self.result['reference_potential_v']:.6g} V | "
			f"Voltage Drop: {self.result['voltage_drop_v'] * 1000.0:.6g} mV | "
			f"Equivalent Resistance: {self.result['resistance_uohm']:.6g} microOhm | "
			f"Power Loss: {self.result['power_loss_w']:.6e} W | "
			f"Conductance: {self.result['conductance_s']:.6e} S\n"
			f"Maximum |J|: {np.max(self.result['actual_current_density_magnitude_a_mm2']):.6e} A/mm^2 | "
			f"Maximum Joule Loss: {np.max(self.result['actual_joule_loss_density']):.6e} W/m^3\n"
			f"Hotspot: X={centroid[0]:.6g} mm, Y={centroid[1]:.6g} mm, Z={centroid[2]:.6g} mm"
		)

	def _current_density_display_range(self, data):
		"""원본 결과 배열은 유지하고 선택된 색상 표시 범위만 계산합니다."""
		if not np.all(np.isfinite(data)):
			raise ValueError("결과 데이터에 NaN 또는 infinite 값이 있습니다.")
		minimum = float(np.min(data))
		maximum = float(np.max(data))
		range_index = self.current_density_range_combo.currentIndex()
		if range_index == 0:
			upper = maximum
			range_name = "Full Range"
		elif range_index in (1, 2, 3, 4, 5):
			percentile = {1: 90, 2: 80, 3: 70, 4: 60, 5: 50}[range_index]
			upper = float(np.percentile(data, percentile))
			range_name = f"{percentile}th Percentile"
		else:
			minimum = self.current_density_min_spin.value()
			upper = self.current_density_max_spin.value()
			range_name = "Custom"
			if upper <= minimum:
				self.current_density_min_spin.setStyleSheet("QDoubleSpinBox { border: 1px solid #c0392b; }")
				self.current_density_max_spin.setStyleSheet("QDoubleSpinBox { border: 1px solid #c0392b; }")
				raise ValueError("Custom Current Density range에서 Max J는 Min J보다 커야 합니다.")
			self.current_density_min_spin.setStyleSheet("")
			self.current_density_max_spin.setStyleSheet("")
		return minimum, upper, range_name

	def _get_grid_result_data(self):
		"""Grid Mode의 결과를 표시 단위에 맞춘 scalar 배열로 반환합니다."""
		characterization_result = self.grid_characterization_result or self.result
		if self.grid_real_battery_result is None and self.result_combo.currentIndex() >= 2:
			raise ValueError("Real Battery Gaussian Model 결과가 아직 준비되지 않았습니다.")
		real_result = self.grid_real_battery_result or self.result
		# Grid FEM은 Lug(Input)을 높은 normalized 전위로 두므로,
		# Lug에서 각 위치까지의 drop은 전체 drop에서 해당 위치 전위를 뺀 값입니다.
		# 복잡한 병렬 frame/wire 경로 때문에 거리가 단순히 저항과 비례하지 않을 수 있습니다.
		lug_voltage_drop = np.abs(
			characterization_result["voltage_drop_v"] - characterization_result["voltage_drop_field_v"]
		)
		index = self.result_combo.currentIndex()
		if index == 0:
			return "Voltage Drop (mV)", lug_voltage_drop * 1000.0, "point", "Voltage Drop (mV)"
		if index == 1:
			return "Resistance Map (microOhm)", lug_voltage_drop / characterization_result["analysis_current_a"] * 1.0e6, "point", "Resistance Map (microOhm)"
		if index == 2:
			return "Real Battery Voltage Drop (mV)", np.abs(real_result["voltage_drop_field_v"]) * 1000.0, "point", "Voltage Drop in Real Battery Model (mV)"
		if index == 3:
			return "Real Battery Current Density (A/mm^2)", real_result["actual_current_density_magnitude_a_mm2"], "cell", "Current Density in Real Battery Model (A/mm^2)"
		return "Real Battery Joule Loss Density (W/m^3)", real_result["actual_joule_loss_density"], "cell", "Joule Loss Density in Real Battery Model (W/m^3)"

	def _build_grid_midplane(self, scalar_name, data, preference):
		"""두께 방향 중앙 좌표에서 Grid 표시용 planar slice를 생성합니다."""
		axis = {"x": 0, "y": 1, "z": 2}.get(self.model_dimensions.get("thickness_axis"))
		if axis is None:
			raise ValueError("Grid thickness axis를 확인할 수 없습니다.")
		bounds = self.grid.bounds
		minimum = bounds[axis * 2]
		maximum = bounds[axis * 2 + 1]
		mid_coordinate = (minimum + maximum) / 2.0
		normal = [0.0, 0.0, 0.0]
		normal[axis] = 1.0
		sliced = self.grid.slice(normal=normal, origin=[
			(bounds[0] + bounds[1]) / 2.0,
			(bounds[2] + bounds[3]) / 2.0,
			(bounds[4] + bounds[5]) / 2.0,
		])
		if sliced.n_points == 0 or sliced.n_cells == 0:
			raise ValueError("Mid-Plane slice가 비어 있습니다.")
		# PyVista slice가 cell scalar를 보존하지 않는 경우를 대비해 원본 mesh의
		# cell data를 임의로 평균하지 않고, 보간된 slice scalar를 사용합니다.
		if preference == "cell" and scalar_name not in sliced.cell_data:
			raise ValueError("Mid-Plane에 cell scalar를 전달하지 못했습니다.")
		return sliced, mid_coordinate

	def _render_grid_display(self):
		"""Grid Mode의 Mid-Plane 및 3D 결과 표시를 갱신합니다."""
		self.voltage_mode_combo.setVisible(False)
		for control in self.current_density_controls:
			if isinstance(control, QLabel):
				control.setVisible(control.text() in ("Range", "Scale"))
			else:
				control.setVisible(control in (
					self.current_density_range_combo,
					self.current_density_scale_combo,
				))
		self.current_density_min_label.setVisible(False)
		self.current_density_max_label.setVisible(False)
		custom_range = self.current_density_range_combo.currentIndex() == 6
		self.hotspot_check.setVisible(self.result_combo.currentIndex() >= 2)
		self.current_density_min_label.setVisible(custom_range)
		self.current_density_max_label.setVisible(custom_range)
		self.current_density_min_spin.setVisible(custom_range)
		self.current_density_max_spin.setVisible(custom_range)
		scalar_name, data, preference, scalar_bar_title = self._get_grid_result_data()
		self.grid.point_data.pop(scalar_name, None)
		self.grid.cell_data.pop(scalar_name, None)
		if preference == "point":
			self.grid.point_data[scalar_name] = data
		else:
			self.grid.cell_data[scalar_name] = data
		vmin, vmax, range_name = self._current_density_display_range(np.asarray(data, dtype=float))
		view_index = self.view_combo.currentIndex()
		log_scale = self.current_density_scale_combo.currentIndex() == 1
		if log_scale:
			positive_data = np.asarray(data)[np.asarray(data) > 0]
			if not len(positive_data):
				raise ValueError("Logarithmic 표시를 위한 양수 scalar가 없습니다.")
			log_floor = max(float(np.min(positive_data)), np.finfo(float).tiny)
			display_data = np.maximum(np.asarray(data, dtype=float), log_floor)
			if preference == "point":
				self.grid.point_data[scalar_name] = display_data
			else:
				self.grid.cell_data[scalar_name] = display_data
			display_min = max(vmin, log_floor)
			display_max = max(vmax, display_min)
		else:
			display_min, display_max = vmin, vmax
		if view_index == 0:
			display_grid, _ = self._build_grid_midplane(scalar_name, data, preference)
		else:
			display_grid = self.grid.extract_surface() if view_index == 1 else self.grid
		self.viewer.add_mesh(display_grid, scalars=scalar_name, preference=preference,
			cmap="viridis", clim=(display_min, display_max), log_scale=log_scale, show_edges=False,
			scalar_bar_args={"title": scalar_bar_title}, pickable=False)
		self.summary_label.setText(
			f"Grid Mode | Model: {'Grid Characterization' if self.result_combo.currentIndex() < 2 else 'Real Battery Gaussian Current Model'} | "
			f"Analysis Current: {self.result['analysis_current_a']:.6g} A | "
			f"Reference Potential: {self.result['reference_potential_v']:.6g} V\n"
			f"Voltage Drop: {self.result['voltage_drop_v'] * 1000.0:.6g} mV | "
			f"Equivalent Resistance: {self.result['resistance_uohm']:.6g} microOhm | "
			f"{scalar_bar_title}\n"
			f"Range: {range_name} | Display: {vmin:.6g} ~ {vmax:.6g}\n"
			"Resistance Map is relative to lug reference, not the total equivalent resistance."
		)
		if self.result_combo.currentIndex() >= 2 and self.grid_real_battery_result is not None:
			real_result = self.grid_real_battery_result
			self.summary_label.setText(
				self.summary_label.text()
				+ f"\nReal Battery Model uses a simplified Gaussian active-material current distribution."
				+ f"\nGaussian Center: {real_result['gaussian_center_xy_mm']} mm | Spread: {real_result['gaussian_spread_xy_mm']} mm"
				+ f"\nIntegrated Sink Current: {real_result['integrated_sink_current_a']:.6g} A | Balance Error: {real_result['sink_current_balance_error_percent']:.6g} %"
			)
		print(f"=== {scalar_bar_title} Statistics ===")
		print(
			f"Min: {np.min(data):.6e} | P50: {np.percentile(data, 50):.6e} | "
			f"P60: {np.percentile(data, 60):.6e} | P70: {np.percentile(data, 70):.6e} | "
			f"P80: {np.percentile(data, 80):.6e} | P90: {np.percentile(data, 90):.6e} | "
			f"Max: {np.max(data):.6e} | Mean: {np.mean(data):.6e}"
		)
		self.viewer.reset_camera()
		if view_index == 0:
			axis = self.model_dimensions.get("thickness_axis")
			view_method = {"x": self.viewer.view_yz, "y": self.viewer.view_xz, "z": self.viewer.view_xy}.get(axis)
			if view_method:
				view_method()
		if view_index != 0 and self.result_combo.currentIndex() == 4 and self.hotspot_check.isChecked():
			position = np.asarray(self.grid_real_battery_result["actual_hotspot"]["centroid_mm"], dtype=float)
			marker = pv.Sphere(radius=max(self.grid.length * 0.01, 1.0), center=position)
			self.viewer.add_mesh(marker, color="red", pickable=False)
		self.viewer.render()

	def _update_display(self):
		"""표시 설정 오류를 대화상자로 알리고 기존 Viewer를 유지합니다."""
		try:
			self._render_display()
		except ValueError as error:
			QMessageBox.warning(self, "Current Density Display", str(error))

	def _render_display(self):
		"""선택된 결과 scalar만 바꾸고 FEM 계산은 다시 실행하지 않습니다."""
		self.viewer.clear()
		if self.grid_mode:
			self._render_grid_display()
			return
		index = self.result_combo.currentIndex()
		self.current_density_min_label.setVisible(False)
		self.current_density_max_label.setVisible(False)
		self.voltage_mode_combo.setVisible(index == 0)
		for control in self.current_density_controls:
			if isinstance(control, QLabel):
				control.setVisible(control.text() in ("Range", "Scale") or (control.text() == "Quantity" and index == 1))
			else:
				control.setVisible(control in (self.current_density_range_combo, self.current_density_scale_combo) or (control is self.current_density_quantity_combo and index == 1))
		self.current_density_min_spin.setVisible(False)
		self.current_density_max_spin.setVisible(False)

		if index == 0 and self.voltage_mode_combo.currentIndex() == 0:
			scalar_name = "Voltage Drop (mV)"
			scalar_bar_title = "Voltage Drop (mV)"
			preference = "point"
			display_data = np.asarray(self.grid.point_data[scalar_name], dtype=float)
			allow_log_scale = True
		elif index == 0:
			scalar_name = "Absolute Potential (V)"
			scalar_bar_title = "Absolute Potential (V)"
			preference = "point"
			display_data = np.asarray(self.grid.point_data[scalar_name], dtype=float)
			allow_log_scale = False
		elif index == 1:
			display_data = np.asarray(self.result["actual_current_density_magnitude_a_mm2"], dtype=float)
			preference = "cell"
			allow_log_scale = True
			if self.current_density_quantity_combo.currentIndex() == 0:
				scalar_name = "Current Density (A/mm^2)"
				scalar_bar_title = "Current Density |J| (A/mm^2)"
				quantity_name = "Absolute J"
			else:
				_, reference_j, _ = self._current_density_display_range(display_data)
				if reference_j <= 0:
					raise ValueError("Normalized J의 reference 값은 0보다 커야 합니다.")
				display_data = display_data / reference_j
				scalar_name = "Normalized Current Density"
				scalar_bar_title = "Normalized Current Density"
				quantity_name = "Normalized J"
		else:
			scalar_name = "Joule Loss Density (W/m^3)"
			scalar_bar_title = "Joule Loss Density (W/m^3)"
			preference = "cell"
			display_data = np.asarray(self.grid.cell_data[scalar_name], dtype=float)
			allow_log_scale = True

		self.current_density_scale_combo.setEnabled(allow_log_scale)
		log_scale = allow_log_scale and self.current_density_scale_combo.currentIndex() == 1
		display_min, display_max, range_name = self._current_density_display_range(display_data)
		if display_max <= display_min:
			display_max = float(np.nextafter(display_min, math.inf))
		if log_scale:
			positive_values = display_data[display_data > 0]
			if not len(positive_values):
				raise ValueError("Logarithmic 표시를 위한 양수 결과 값이 없습니다.")
			log_floor = max(float(np.min(positive_values)), np.finfo(float).tiny)
			display_data = np.maximum(display_data, log_floor)
			display_min = max(display_min, log_floor)
			display_max = max(display_max, float(np.nextafter(display_min, math.inf)))
		if preference == "point":
			self.grid.point_data[scalar_name] = display_data
		else:
			self.grid.cell_data[scalar_name] = display_data

		self._update_summary()
		scale_name = "Logarithmic" if log_scale else "Linear"
		extra_summary = f"\nRange: {range_name} | Scale: {scale_name} | Display Range: {display_min:.6g} ~ {display_max:.6g}"
		if index == 1:
			extra_summary += f" | Quantity: {quantity_name}"
		if index == 0 and not allow_log_scale:
			extra_summary += "\nAbsolute Potential은 0 이하 값을 포함할 수 있어 Linear scale만 사용합니다."
		self.summary_label.setText(self.summary_label.text() + extra_summary)

		display_grid = self.grid.extract_surface()
		self.viewer.add_mesh(
			display_grid,
			scalars=scalar_name,
			preference=preference,
			show_edges=False,
			cmap="viridis",
			clim=(display_min, display_max),
			log_scale=log_scale,
			scalar_bar_args={"title": scalar_bar_title},
			pickable=False,
		)
		if index == 2 and self.hotspot_check.isChecked():
			hotspot = self.result["actual_hotspot"]
			position = np.asarray(hotspot["centroid_mm"], dtype=float)
			marker = pv.Sphere(radius=max(self.grid.length * 0.01, 1.0), center=position)
			self.viewer.add_mesh(marker, color="red", pickable=False)
			self.viewer.add_point_labels(
				position.reshape(1, 3),
				["Hotspot"],
				point_size=1,
				font_size=12,
				shape=None,
				pickable=False,
			)
		self.viewer.reset_camera()
		self.viewer.render()

	def closeEvent(self, event):
		"""결과 창만 닫고 solver 결과와 Gmsh 모델은 유지합니다."""
		parent = self.parent()
		self.viewer.close()
		if parent is not None and getattr(parent, "fem_results_dialog", None) is self:
			parent.fem_results_dialog = None
		event.accept()


class SurfaceSelectionWindow(QMainWindow):
	"""Surface를 클릭하고 Input/Output 집합을 관리하는 메인 창입니다."""

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Lead-Acid Battery Grid Resistance Analyzer")
		self.resize(1200, 720)

		# 현재 경계 선택 모드입니다. 1은 Input만 선택하고,
		# 2는 Input과 Output을 따로 선택합니다.
		self.boundary_mode = 1
		self.selection_type = "input"
		self.input_surfaces = set()
		self.output_surfaces = set()
		self.all_surfaces = set()
		self.actor_by_surface_id = {}
		self.surface_id_by_actor_key = {}
		self.current_step_path = None
		self.model_dimensions = None
		self.material_name = "Pb-Ca-Sn Alloy"
		self.materials = self._load_material_library()
		self.analysis_config = None
		self.fem_result = None
		self.normalized_fem_result = None
		self.grid_characterization_result = None
		self.grid_real_battery_result = None
		self.mesh_preview_dialog = None
		self.fem_results_dialog = None

		self._build_toolbar()
		self._build_status_panel()
		self._build_control_panel()
		self._build_menu_bar()
		self._set_empty_status()
		if AUTO_LOAD_DEFAULT_STEP:
			default_step_path = Path(__file__).resolve().parent / "grid.step"
			if default_step_path.exists():
				self._load_step_and_build_viewer(default_step_path)
		self._update_controls()

	def _get_materials_file_path(self):
		"""설치 폴더가 아닌 사용자별 AppData의 JSON 경로를 반환합니다."""
		app_data_path = Path(
			QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
		)
		app_data_path.mkdir(parents=True, exist_ok=True)
		return app_data_path / "materials.json"

	def _default_material_library(self):
		"""파일이 없거나 복구할 때 사용할 최초 Material 목록입니다."""
		return [
			{
				"name": "Pb-Ca-Sn Alloy",
				"resistivity": DEFAULT_RESISTIVITY,
				"reference_temperature": DEFAULT_REFERENCE_TEMPERATURE,
			}
		]

	def _is_valid_material_library(self, materials):
		"""JSON 자료가 현재 Material record 형식인지 확인합니다."""
		if not isinstance(materials, list) or not materials:
			return False
		seen_names = set()
		for material in materials:
			if not isinstance(material, dict):
				return False
			name = material.get("name")
			try:
				resistivity = float(material["resistivity"])
				float(material["reference_temperature"])
			except (KeyError, TypeError, ValueError):
				return False
			name_key = name.strip().casefold() if isinstance(name, str) else ""
			if not name_key or name_key in seen_names or resistivity <= 0:
				return False
			seen_names.add(name_key)
		return True

	def _load_material_library(self):
		"""materials.json을 읽고, 문제가 있으면 기본 목록으로 계속 실행합니다."""
		materials_path = self._get_materials_file_path()
		if not materials_path.exists():
			materials = self._default_material_library()
			self._save_material_library(materials)
			return materials
		try:
			with materials_path.open("r", encoding="utf-8") as material_file:
				materials = json.load(material_file)
			if not self._is_valid_material_library(materials):
				raise ValueError("Invalid material record")
			return materials
		except (OSError, json.JSONDecodeError, ValueError, TypeError):
			reply = QMessageBox.warning(
			self,
			"Material Library Error",
			"Material Library를 읽을 수 없습니다. 기본 library로 복구하시겠습니까?",
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
		)
		materials = self._default_material_library()
		if reply == QMessageBox.StandardButton.Yes:
			self._save_material_library(materials)
		return materials

	def _save_material_library(self, materials=None):
		"""현재 Material 목록을 사용자 AppData의 JSON 파일에 저장합니다."""
		materials = self.materials if materials is None else materials
		with self._get_materials_file_path().open("w", encoding="utf-8") as material_file:
			json.dump(materials, material_file, indent=4, ensure_ascii=False)

	def _refresh_material_combo(self, selected_name=None):
		"""Library 변경 후 메인 ComboBox를 다시 채우고 선택을 유지합니다."""
		selected_name = selected_name or self.material_combo.currentText()
		self.material_combo.blockSignals(True)
		self.material_combo.clear()
		self.material_combo.addItems([material["name"] for material in self.materials])
		index = self.material_combo.findText(selected_name, Qt.MatchFlag.MatchExactly)
		self.material_combo.setCurrentIndex(index if index >= 0 else 0)
		self.material_combo.blockSignals(False)
		self._on_material_changed(self.material_combo.currentText())

	def _build_menu_bar(self):
		"""Toolbar와 같은 함수를 호출하는 상단 메뉴를 만듭니다."""
		file_menu = self.menuBar().addMenu("File")
		open_action = file_menu.addAction("Open STEP...")
		open_action.triggered.connect(self.open_step_file)
		close_action = file_menu.addAction("Close Model")
		close_action.triggered.connect(self.close_model)
		file_menu.addSeparator()
		exit_action = file_menu.addAction("Exit")
		exit_action.triggered.connect(self.close)

		view_menu = self.menuBar().addMenu("View")
		isometric_action = view_menu.addAction("Isometric View")
		isometric_action.triggered.connect(self.set_isometric_view)
		top_action = view_menu.addAction("Top View")
		top_action.triggered.connect(self.set_top_view)
		bottom_action = view_menu.addAction("Bottom View")
		bottom_action.triggered.connect(self.set_bottom_view)
		front_action = view_menu.addAction("Front View")
		front_action.triggered.connect(self.set_front_view)
		back_action = view_menu.addAction("Back View")
		back_action.triggered.connect(self.set_back_view)
		right_action = view_menu.addAction("Right View")
		right_action.triggered.connect(self.set_right_view)
		left_action = view_menu.addAction("Left View")
		left_action.triggered.connect(self.set_left_view)
		view_menu.addSeparator()
		reset_camera_action = view_menu.addAction("Reset Camera")
		reset_camera_action.triggered.connect(self.reset_camera)

		boundary_menu = self.menuBar().addMenu("Boundary")
		boundary_menu.addAction("Grid", lambda: self.set_boundary_mode(1))
		boundary_menu.addAction("COS/Strap", lambda: self.set_boundary_mode(2))
		boundary_menu.addSeparator()
		boundary_menu.addAction("Select Input", lambda: self.set_selection_type("input"))
		boundary_menu.addAction("Select Output", lambda: self.set_selection_type("output"))
		boundary_menu.addSeparator()
		boundary_menu.addAction("Confirm", self.confirm_selection)
		boundary_menu.addAction("Clear All", self.clear_selection)

		help_menu = self.menuBar().addMenu("Help")
		help_menu.addAction("About", self.show_about)

	def _build_toolbar(self):
		"""상단 toolbar와 버튼을 만듭니다."""
		toolbar = QToolBar("Boundary controls", self)
		toolbar.setMovable(False)
		self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

		self.open_step_button = QPushButton("Open STEP")
		self.open_step_button.clicked.connect(self.open_step_file)
		toolbar.addWidget(self.open_step_button)
		toolbar.addSeparator()

		self.option_1_action = QAction("Grid", self)
		self.option_1_action.setCheckable(True)
		self.option_1_action.triggered.connect(lambda: self.set_boundary_mode(1))
		toolbar.addAction(self.option_1_action)

		self.option_2_action = QAction("COS/Strap", self)
		self.option_2_action.setCheckable(True)
		self.option_2_action.triggered.connect(lambda: self.set_boundary_mode(2))
		toolbar.addAction(self.option_2_action)

		toolbar.addSeparator()
		self.input_button = QPushButton("Select Input")
		self.input_button.setCheckable(True)
		self.input_button.clicked.connect(lambda: self.set_selection_type("input"))
		toolbar.addWidget(self.input_button)

		self.output_button = QPushButton("Select Output")
		self.output_button.setCheckable(True)
		self.output_button.clicked.connect(lambda: self.set_selection_type("output"))
		toolbar.addWidget(self.output_button)

		toolbar.addSeparator()
		self.confirm_button = QPushButton("Confirm")
		self.confirm_button.clicked.connect(self.confirm_selection)
		toolbar.addWidget(self.confirm_button)

		self.clear_button = QPushButton("Clear All")
		self.clear_button.clicked.connect(self.clear_selection)
		toolbar.addWidget(self.clear_button)

		toolbar.addSeparator()
		self.toolbar_generate_mesh_button = QPushButton("Generate 3D Mesh")
		self.toolbar_generate_mesh_button.clicked.connect(self.generate_3d_mesh)
		self.toolbar_generate_mesh_button.setEnabled(False)
		toolbar.addWidget(self.toolbar_generate_mesh_button)
		self.toolbar_run_fem_button = QPushButton("Run FEM Analysis")
		self.toolbar_run_fem_button.clicked.connect(self.run_fem_analysis)
		self.toolbar_run_fem_button.setEnabled(False)
		toolbar.addWidget(self.toolbar_run_fem_button)
		self.toolbar_view_fem_results_button = QPushButton("View FEM Results")
		self.toolbar_view_fem_results_button.clicked.connect(self._open_fem_results)
		self.toolbar_view_fem_results_button.setEnabled(False)
		toolbar.addWidget(self.toolbar_view_fem_results_button)

	def _build_status_panel(self):
		"""Viewer 위에 모델 정보, 현재 모드와 Surface 개수를 표시합니다."""
		self.model_info_label = QLabel()
		self.model_info_label.setStyleSheet(
			"QLabel { padding: 6px 8px; background: #e8f1f8; font-size: 13px; }"
		)
		self.status_label = QLabel()
		self.status_label.setStyleSheet(
			"QLabel { padding: 8px; background: #f0f0f0; font-size: 14px; }"
		)
		# Viewer의 카메라 조작 방법을 영문으로 표시합니다.
		self.view_help_label = QLabel(
			"View controls: Left drag = Rotate | Right drag = Zoom | "
			"Shift + Left drag = Pan<br>"
			"Red = Current Input | Blue = Output"
		)
		self.view_help_label.setStyleSheet(
			"QLabel { padding: 5px 8px; color: #555; background: #fafafa; }"
		)

		self.viewer = QtInteractor(self)
		self.viewer_area = QWidget(self)
		layout = QVBoxLayout(self.viewer_area)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self.model_info_label)
		layout.addWidget(self.status_label)
		layout.addWidget(self.view_help_label)
		layout.addWidget(self.viewer, stretch=1)

	def _build_control_panel(self):
		"""Boundary, Material, Mesh 설정을 담은 좌측 스크롤 패널을 만듭니다."""
		self.control_panel = QWidget()
		self.control_panel.setMinimumWidth(380)
		control_layout = QVBoxLayout(self.control_panel)
		control_layout.setContentsMargins(8, 8, 8, 8)
		control_layout.addWidget(self._build_boundary_group())
		control_layout.addWidget(self._build_material_group())
		control_layout.addWidget(self._build_mesh_group())
		control_layout.addWidget(self._build_fem_group())
		# 설정 항목이 늘어나도 아래쪽 여유 공간이 유지되도록 합니다.
		control_layout.addStretch(1)

		scroll_area = QScrollArea(self)
		scroll_area.setWidgetResizable(True)
		scroll_area.setMinimumWidth(400)
		scroll_area.setWidget(self.control_panel)

		splitter = QSplitter(Qt.Orientation.Horizontal, self)
		splitter.addWidget(scroll_area)
		splitter.addWidget(self.viewer_area)
		splitter.setSizes([420, 780])
		splitter.setStretchFactor(0, 0)
		splitter.setStretchFactor(1, 1)
		self.setCentralWidget(splitter)

	def _build_boundary_group(self):
		"""기존 Boundary 함수를 호출하는 좌측 설정 영역을 만듭니다."""
		group = QGroupBox("Boundary Conditions")
		layout = QVBoxLayout(group)

		mode_label = QLabel("Boundary Mode")
		layout.addWidget(mode_label)
		self.panel_option_1 = QRadioButton("Grid")
		self.panel_option_1.setToolTip("Input only / Remaining surfaces = Output")
		self.panel_option_1.clicked.connect(lambda: self.set_boundary_mode(1))
		layout.addWidget(self.panel_option_1)
		self.panel_option_2 = QRadioButton("COS/Strap")
		self.panel_option_2.setToolTip("Manual Input / Output")
		self.panel_option_2.clicked.connect(lambda: self.set_boundary_mode(2))
		layout.addWidget(self.panel_option_2)

		self.panel_input_button = QPushButton("Select Input")
		self.panel_input_button.setCheckable(True)
		self.panel_input_button.clicked.connect(
			lambda: self.set_selection_type("input")
		)
		layout.addWidget(self.panel_input_button)
		self.panel_output_button = QPushButton("Select Output")
		self.panel_output_button.setCheckable(True)
		self.panel_output_button.clicked.connect(
			lambda: self.set_selection_type("output")
		)
		layout.addWidget(self.panel_output_button)

		self.panel_boundary_status = QLabel()
		self.panel_boundary_status.setWordWrap(True)
		self.panel_boundary_status.setStyleSheet("QLabel { color: #555; padding: 4px; }")
		layout.addWidget(self.panel_boundary_status)

		button_layout = QHBoxLayout()
		self.panel_confirm_button = QPushButton("Confirm")
		self.panel_confirm_button.clicked.connect(self.confirm_selection)
		button_layout.addWidget(self.panel_confirm_button)
		self.panel_clear_button = QPushButton("Clear All")
		self.panel_clear_button.clicked.connect(self.clear_selection)
		button_layout.addWidget(self.panel_clear_button)
		layout.addLayout(button_layout)
		return group

	def _build_material_group(self):
		"""나중에 해석기가 읽을 Material 기본 입력 영역을 만듭니다."""
		group = QGroupBox("Material Properties")
		layout = QFormLayout(group)

		self.material_combo = QComboBox()
		self.material_combo.addItems([material["name"] for material in self.materials])
		self.material_combo.currentTextChanged.connect(self._on_material_changed)
		layout.addRow("Material", self.material_combo)

		self.manage_materials_button = QPushButton("Manage Materials...")
		self.manage_materials_button.clicked.connect(self._open_material_manager)
		layout.addRow(self.manage_materials_button)

		self.resistivity_edit = QLineEdit()
		self.resistivity_edit.setText(f"{DEFAULT_RESISTIVITY:.2e}")
		resistivity_validator = QDoubleValidator(
			0.0, 1.0e100, 16, self.resistivity_edit
		)
		resistivity_validator.setNotation(
			QDoubleValidator.Notation.ScientificNotation
		)
		self.resistivity_edit.setValidator(resistivity_validator)
		self.resistivity_edit.textChanged.connect(self._on_resistivity_changed)
		self.resistivity_edit.editingFinished.connect(self._validate_resistivity)
		layout.addRow("Resistivity (Ohm m)", self.resistivity_edit)

		self.conductivity_edit = QLineEdit()
		self.conductivity_edit.setReadOnly(True)
		layout.addRow("Electrical Conductivity (S/m)", self.conductivity_edit)

		self.reference_temperature_edit = QLineEdit("20.0")
		self.reference_temperature_edit.setValidator(
			QDoubleValidator(-1.0e100, 1.0e100, 8, self.reference_temperature_edit)
		)
		self.reference_temperature_edit.textChanged.connect(self._invalidate_mesh)
		layout.addRow("Reference Temperature (C)", self.reference_temperature_edit)
		self._update_conductivity()
		return group

	def _build_mesh_group(self):
		"""3D Mesh 생성에 필요한 두께 요소 수와 상태를 표시합니다."""
		group = QGroupBox("Mesh / Analysis Preparation")
		layout = QFormLayout(group)

		self.thickness_elements_spin = NoWheelSpinBox()
		self.thickness_elements_spin.setRange(1, 10)
		self.thickness_elements_spin.setValue(2)
		self.thickness_elements_spin.valueChanged.connect(self._on_mesh_setting_changed)
		layout.addRow("Thickness Elements", self.thickness_elements_spin)

		self.target_mesh_size_label = QLabel("N/A")
		layout.addRow("Target Mesh Size (mm)", self.target_mesh_size_label)

		self.generate_mesh_button = QPushButton("Generate 3D Mesh")
		self.generate_mesh_button.clicked.connect(self.generate_3d_mesh)
		self.generate_mesh_button.setEnabled(False)
		layout.addRow(self.generate_mesh_button)

		self.preview_mesh_button = QPushButton("Preview Mesh")
		self.preview_mesh_button.clicked.connect(self._open_mesh_preview)
		self.preview_mesh_button.setEnabled(False)
		layout.addRow(self.preview_mesh_button)

		self.mesh_status_label = QLabel("Mesh Status: Not generated")
		self.mesh_status_label.setWordWrap(True)
		layout.addRow(self.mesh_status_label)
		return group

	def _build_fem_group(self):
		"""FEM 실행 버튼과 숫자 결과를 표시하는 좌측 영역을 만듭니다."""
		group = QGroupBox("FEM Analysis")
		layout = QFormLayout(group)

		self.run_fem_button = QPushButton("Run FEM Analysis")
		self.run_fem_button.clicked.connect(self.run_fem_analysis)
		self.run_fem_button.setEnabled(False)
		layout.addRow(self.run_fem_button)
		self.view_fem_results_button = QPushButton("View FEM Results")
		self.view_fem_results_button.clicked.connect(self._open_fem_results)
		self.view_fem_results_button.setEnabled(False)
		layout.addRow(self.view_fem_results_button)
		self.copy_fem_results_button = QPushButton("Copy Analysis Results")
		self.copy_fem_results_button.clicked.connect(self._copy_fem_results)
		self.copy_fem_results_button.setEnabled(False)
		layout.addRow(self.copy_fem_results_button)

		self.fem_status_label = QLabel("FEM Status: Not solved")
		self.fem_status_label.setWordWrap(True)
		layout.addRow(self.fem_status_label)
		self.analysis_current_spin = NoWheelDoubleSpinBox()
		self.analysis_current_spin.setRange(0.001, 1000000.0)
		self.analysis_current_spin.setDecimals(3)
		self.analysis_current_spin.setValue(100.0)
		self.analysis_current_spin.valueChanged.connect(self._on_analysis_current_changed)
		layout.addRow("Analysis Current (A)", self.analysis_current_spin)
		self.gaussian_spread_x_spin = NoWheelDoubleSpinBox()
		self.gaussian_spread_x_spin.setRange(0.001, 1.0e6)
		self.gaussian_spread_x_spin.setValue(10.0)
		self.gaussian_spread_x_spin.setSuffix(" mm")
		self.gaussian_spread_x_spin.valueChanged.connect(self._invalidate_real_battery_result)
		layout.addRow("Gaussian Spread X", self.gaussian_spread_x_spin)
		self.gaussian_spread_y_spin = NoWheelDoubleSpinBox()
		self.gaussian_spread_y_spin.setRange(0.001, 1.0e6)
		self.gaussian_spread_y_spin.setValue(10.0)
		self.gaussian_spread_y_spin.setSuffix(" mm")
		self.gaussian_spread_y_spin.valueChanged.connect(self._invalidate_real_battery_result)
		layout.addRow("Gaussian Spread Y", self.gaussian_spread_y_spin)
		self.reference_potential_spin = NoWheelDoubleSpinBox()
		self.reference_potential_spin.setRange(-1000.0, 1000.0)
		self.reference_potential_spin.setDecimals(6)
		self.reference_potential_spin.setValue(2.14)
		self.reference_potential_spin.valueChanged.connect(self._on_reference_potential_changed)
		layout.addRow("Reference Potential (V)", self.reference_potential_spin)
		self.fem_min_potential_label = QLabel("N/A")
		self.fem_max_potential_label = QLabel("N/A")
		self.fem_resistance_label = QLabel("N/A")
		self.fem_conductance_label = QLabel("N/A")
		self.fem_current_label = QLabel("N/A")
		self.fem_voltage_drop_label = QLabel("N/A")
		self.fem_power_loss_label = QLabel("N/A")
		self.fem_max_current_density_label = QLabel("N/A")
		self.fem_max_joule_loss_label = QLabel("N/A")
		self.fem_input_potential_label = QLabel("N/A")
		self.fem_output_potential_label = QLabel("N/A")
		layout.addRow("Normalized Potential Min", self.fem_min_potential_label)
		layout.addRow("Normalized Potential Max", self.fem_max_potential_label)
		layout.addRow("Resistance (microOhm)", self.fem_resistance_label)
		layout.addRow("Conductance (S)", self.fem_conductance_label)
		layout.addRow("Current @ 1 V (A)", self.fem_current_label)
		layout.addRow("Voltage Drop (V)", self.fem_voltage_drop_label)
		layout.addRow("Input Potential (V)", self.fem_input_potential_label)
		layout.addRow("Output Potential (V)", self.fem_output_potential_label)
		layout.addRow("Power Loss (W)", self.fem_power_loss_label)
		layout.addRow("Maximum Current Density (A/mm^2)", self.fem_max_current_density_label)
		layout.addRow("Maximum Joule Loss (W/m^3)", self.fem_max_joule_loss_label)
		return group

	def _on_resistivity_changed(self):
		"""비저항을 직접 바꾸면 전도도를 갱신하고 기존 Mesh를 무효화합니다."""
		self._update_conductivity()
		self._invalidate_mesh()

	def _on_mesh_setting_changed(self):
		"""두께 요소 수가 바뀌면 목표 크기와 Mesh 상태를 갱신합니다."""
		self._update_target_mesh_size()
		self._invalidate_mesh()

	def _update_target_mesh_size(self):
		"""현재 STEP 두께를 두께 방향 요소 수로 나눈 목표 크기를 표시합니다."""
		if self.model_dimensions is None:
			self.target_mesh_size_label.setText("N/A")
			return None
		mesh_size = self.model_dimensions["thickness"] / self.thickness_elements_spin.value()
		self.target_mesh_size_label.setText(f"{mesh_size:.6g}")
		return mesh_size

	def _sync_analysis_toolbar_buttons(self):
		"""좌측 분석 버튼의 활성 상태를 상단 toolbar 버튼에 반영합니다."""
		if hasattr(self, "toolbar_generate_mesh_button") and hasattr(self, "generate_mesh_button"):
			self.toolbar_generate_mesh_button.setEnabled(self.generate_mesh_button.isEnabled())
		if hasattr(self, "toolbar_run_fem_button") and hasattr(self, "run_fem_button"):
			self.toolbar_run_fem_button.setEnabled(self.run_fem_button.isEnabled())
		if hasattr(self, "toolbar_view_fem_results_button") and hasattr(self, "view_fem_results_button"):
			self.toolbar_view_fem_results_button.setEnabled(self.view_fem_results_button.isEnabled())

	def _invalidate_mesh(self):
		"""Geometry, 경계, Material, Mesh 설정이 바뀌면 분석 준비 결과를 폐기합니다."""
		if not hasattr(self, "analysis_config"):
			return
		self.analysis_config = None
		if hasattr(self, "mesh_status_label"):
			self.mesh_status_label.setText("Mesh Status: Not generated")
		if hasattr(self, "preview_mesh_button"):
			self.preview_mesh_button.setEnabled(False)
		self.fem_result = None
		self.grid_characterization_result = None
		self.grid_real_battery_result = None
		if hasattr(self, "view_fem_results_button"):
			self.view_fem_results_button.setEnabled(False)
		if hasattr(self, "copy_fem_results_button"):
			self.copy_fem_results_button.setEnabled(False)
		if hasattr(self, "run_fem_button"):
			self.run_fem_button.setEnabled(False)
			self.fem_status_label.setText("FEM Status: Not solved")
			for label in (
				self.fem_min_potential_label,
				self.fem_max_potential_label,
				self.fem_resistance_label,
				self.fem_conductance_label,
				self.fem_current_label,
				self.fem_voltage_drop_label,
				self.fem_power_loss_label,
				self.fem_max_current_density_label,
				self.fem_max_joule_loss_label,
				self.fem_input_potential_label,
				self.fem_output_potential_label,
			):
				label.setText("N/A")
		if self.mesh_preview_dialog is not None:
			self.mesh_preview_dialog.close()
			self.mesh_preview_dialog = None
		if self.fem_results_dialog is not None:
			self.fem_results_dialog.close()
			self.fem_results_dialog = None
		self._sync_analysis_toolbar_buttons()

	def _remove_physical_groups(self):
		"""이전 생성 결과의 Physical Group만 제거하고 Geometry는 보존합니다."""
		physical_groups = gmsh.model.getPhysicalGroups()
		if physical_groups:
			gmsh.model.removePhysicalGroups(physical_groups)

	def _find_grid_measurement_surfaces(self):
		"""Grid Mode에서 Lug 반대편의 far-side measurement Surface를 찾습니다."""
		if not self.input_surfaces:
			return set()
		lug_centers = []
		for surface_id in self.input_surfaces:
			box = gmsh.model.getBoundingBox(2, surface_id)
			lug_centers.append(np.array([(box[0] + box[3]) / 2.0, (box[1] + box[4]) / 2.0, (box[2] + box[5]) / 2.0]))
		lug_center = np.mean(lug_centers, axis=0)
		surface_distances = []
		for surface_id in self.all_surfaces - self.input_surfaces:
			box = gmsh.model.getBoundingBox(2, surface_id)
			center = np.array([(box[0] + box[3]) / 2.0, (box[1] + box[4]) / 2.0, (box[2] + box[5]) / 2.0])
			surface_distances.append((float(np.linalg.norm(center - lug_center)), surface_id))
		if not surface_distances:
			return set()
		maximum_distance = max(distance for distance, _ in surface_distances)
		# 같은 far-side 평면에 속한 Surface가 여러 개일 수 있어 상대 오차로 묶습니다.
		return {
			surface_id
			for distance, surface_id in surface_distances
			if math.isclose(distance, maximum_distance, rel_tol=1.0e-6, abs_tol=1.0e-9)
		}

	def _count_mesh_elements(self):
		"""현재 Gmsh mesh에서 3D 전체 요소와 1차 tetra 개수를 셉니다."""
		element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
		total_elements = sum(len(tags) for tags in element_tags)
		tetrahedra = 0
		for element_type, tags in zip(element_types, element_tags):
			_, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(
				element_type
			)
			if nodes_per_element == 4:
				tetrahedra += len(tags)
		return total_elements, tetrahedra

	def _build_pyvista_volume_mesh(self):
		"""현재 Gmsh의 4-node tetra만 PyVista UnstructuredGrid로 변환합니다."""
		node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
		points = np.asarray(coordinates, dtype=float).reshape((-1, 3))
		node_index_by_tag = {
			int(node_tag): index for index, node_tag in enumerate(node_tags)
		}
		element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(3)
		cells = []
		tetrahedron_count = 0
		for element_type, tags, node_tags_for_elements in zip(
			element_types, element_tags, element_node_tags
		):
			_, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(
				element_type
			)
			if nodes_per_element != 4:
				continue
			elements = np.asarray(node_tags_for_elements, dtype=np.int64).reshape((-1, 4))
			for element in elements:
				try:
					cells.extend([4, *(node_index_by_tag[int(tag)] for tag in element)])
				except KeyError as error:
					raise ValueError(f"Gmsh node mapping failed for tag {error.args[0]}.") from error
			tetrahedron_count += len(tags)
		if not tetrahedron_count:
			raise ValueError("현재 Mesh에 4-node tetrahedral element가 없습니다.")
		cell_array = np.asarray(cells, dtype=np.int64)
		cell_types = np.full(tetrahedron_count, pv.CellType.TETRA, dtype=np.uint8)
		return pv.UnstructuredGrid(cell_array, cell_types, points), tetrahedron_count, len(node_tags)

	def _open_mesh_preview(self):
		"""현재 Mesh를 별도 Preview 창에 표시하고 기존 창은 재사용합니다."""
		if self.analysis_config is None:
			QMessageBox.warning(self, "Mesh Preview", "먼저 3D Mesh를 생성해 주세요.")
			return
		try:
			mesh_grid, tetrahedron_count, node_count = self._build_pyvista_volume_mesh()
			if tetrahedron_count >= MESH_PREVIEW_WARNING_ELEMENTS:
				reply = QMessageBox.question(
					self,
					"Large Mesh",
					"Mesh is very large. Preview may be slow. Continue?",
					QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
				)
				if reply != QMessageBox.StandardButton.Yes:
					return
			if self.mesh_preview_dialog is not None:
				self.mesh_preview_dialog.close()
			self.mesh_preview_dialog = MeshPreviewDialog(
				self, mesh_grid, self.analysis_config, node_count, tetrahedron_count
			)
			self.mesh_preview_dialog.show()
			self.mesh_preview_dialog.raise_()
			self.mesh_preview_dialog.activateWindow()
		except Exception as error:
			QMessageBox.critical(self, "Mesh Preview Error", str(error))

	def generate_3d_mesh(self):
		"""선택된 경계와 Material을 검증한 뒤 3D tetrahedral Mesh를 생성합니다."""
		if self.current_step_path is None or self.model_dimensions is None:
			QMessageBox.warning(self, "Mesh Generation", "먼저 STEP 파일을 열어 주세요.")
			return
		if not self.input_surfaces:
			QMessageBox.warning(self, "Boundary Validation", "Input Surface를 하나 이상 선택해 주세요.")
			return
		if self.boundary_mode == 1:
			# Grid는 Lug와 전체 외부면 사이의 분산 전류 모델이 아닙니다.
			# Lug 반대편의 far-side measurement terminal을 Output으로 사용합니다.
			output_surfaces = self._find_grid_measurement_surfaces()
			insulated_surfaces = set()
		else:
			output_surfaces = set(self.output_surfaces)
			insulated_surfaces = self.all_surfaces - self.input_surfaces - output_surfaces
		if self.boundary_mode == 2 and not output_surfaces:
			QMessageBox.warning(self, "Boundary Validation", "Output Surface를 하나 이상 선택해 주세요.")
			return
		if not output_surfaces:
			QMessageBox.warning(self, "Boundary Validation", "Output Surface를 만들 수 없습니다.")
			return

		material = self.get_material_properties()
		if (
			material["resistivity"] is None
			or material["conductivity"] is None
			or material["reference_temperature"] is None
			or not math.isfinite(material["resistivity"])
			or not math.isfinite(material["conductivity"])
			or material["resistivity"] <= 0
			or material["conductivity"] <= 0
			or not math.isfinite(material["reference_temperature"])
		):
			QMessageBox.warning(self, "Material Validation", "Material properties가 유효하지 않습니다.")
			return

		mesh_size = self._update_target_mesh_size()
		thickness_elements = self.thickness_elements_spin.value()
		progress_dialog = QProgressDialog("Preparing Physical Groups...", None, 0, 3, self)
		progress_dialog.setWindowTitle("Generating 3D Mesh")
		progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
		progress_dialog.setMinimumDuration(0)
		progress_dialog.setValue(0)
		self.mesh_status_label.setText("Mesh Status: Generating...")
		QApplication.processEvents()
		try:
			# 기존 Group만 지우므로 STEP Geometry와 Surface ID는 그대로 유지됩니다.
			self._remove_physical_groups()
			input_physical_tag = gmsh.model.addPhysicalGroup(2, sorted(self.input_surfaces))
			gmsh.model.setPhysicalName(2, input_physical_tag, "Input")
			output_physical_tag = gmsh.model.addPhysicalGroup(2, sorted(output_surfaces))
			gmsh.model.setPhysicalName(2, output_physical_tag, "Output")
			volume_entities = gmsh.model.getEntities(3)
			if not volume_entities:
				raise ValueError("3D Volume entity를 찾을 수 없습니다.")
			volume_tags = [tag for _, tag in volume_entities]
			grid_physical_tag = gmsh.model.addPhysicalGroup(3, volume_tags)
			gmsh.model.setPhysicalName(3, grid_physical_tag, "Grid")
			progress_dialog.setValue(1)
			progress_dialog.setLabelText("Generating 3D Volume Mesh...")
			QApplication.processEvents()

			# 2D Viewer용 mesh를 비우고 같은 Geometry에서 3D mesh를 생성합니다.
			gmsh.model.mesh.clear()
			gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
			gmsh.option.setNumber("Mesh.ElementOrder", 1)
			gmsh.model.mesh.generate(3)
			progress_dialog.setValue(2)
			progress_dialog.setLabelText("Reading Mesh Information...")
			QApplication.processEvents()

			node_tags, _, _ = gmsh.model.mesh.getNodes()
			total_3d_elements, tetrahedral_count = self._count_mesh_elements()
			if tetrahedral_count == 0:
				raise ValueError("생성된 Mesh에 tetrahedral element가 없습니다.")
			self.analysis_config = {
				"step_file": str(self.current_step_path),
				"boundary_mode": self.boundary_mode,
				"input_surfaces": sorted(self.input_surfaces),
				"output_surfaces": sorted(output_surfaces),
				"insulated_surfaces": sorted(insulated_surfaces),
				"material": material,
				"mesh": {
					"thickness_elements": thickness_elements,
					"target_mesh_size_mm": mesh_size,
				},
				"physical_groups": {
					"input": input_physical_tag,
					"output": output_physical_tag,
					"grid": grid_physical_tag,
				},
			}
			self.mesh_status_label.setText(
				f"Mesh Status: Ready\nNodes: {len(node_tags)}\nTetrahedra: {tetrahedral_count}"
			)
			self.preview_mesh_button.setEnabled(True)
			self.run_fem_button.setEnabled(True)
			self._sync_analysis_toolbar_buttons()
			progress_dialog.setValue(3)
			progress_dialog.setLabelText("Finished")
			print("\n=== 3D Mesh / Boundary Preparation ===")
			print(f"STEP: {self.current_step_path}")
			print(f"Material: {material['material']}")
			print(f"Resistivity: {material['resistivity']:.6e} Ohm m")
			print(f"Conductivity: {material['conductivity']:.6e} S/m")
			print(f"Thickness: {self.model_dimensions['thickness']:.3f} mm")
			print(f"Thickness Elements: {thickness_elements}")
			print(f"Target Mesh Size: {mesh_size:.6g} mm")
			print(f"Nodes: {len(node_tags)}")
			print(f"3D Elements: {total_3d_elements}")
			print(f"Tetrahedral Elements: {tetrahedral_count}")
			print(f"Input Physical Group: Tag {input_physical_tag}, Surfaces: {len(self.input_surfaces)}")
			print(f"Output Physical Group: Tag {output_physical_tag}, Surfaces: {len(output_surfaces)}")
			print(f"Grid Physical Volume: Tag {grid_physical_tag}")
		except Exception as error:
			self.analysis_config = None
			self.mesh_status_label.setText("Mesh Status: Failed")
			self.preview_mesh_button.setEnabled(False)
			QMessageBox.critical(self, "Mesh Generation Error", str(error))
		finally:
			self._sync_analysis_toolbar_buttons()
			progress_dialog.close()

	def _load_step_and_build_viewer(self, step_path):
		"""기존 모델을 정리하고 지정한 STEP를 Viewer에 표시합니다."""
		self._clear_model_data()
		progress_dialog = QProgressDialog("Preparing STEP load...", None, 0, 100, self)
		progress_dialog.setWindowTitle("Loading STEP")
		progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
		progress_dialog.setMinimumDuration(0)
		progress_dialog.setValue(0)
		QApplication.processEvents()
		try:
			if not gmsh.isInitialized():
				gmsh.initialize()
			gmsh.option.setNumber("General.Terminal", 1)
			gmsh.clear()
			gmsh.model.add("grid_from_step")
			progress_dialog.setLabelText("Importing STEP geometry...")
			progress_dialog.setValue(10)
			QApplication.processEvents()
			gmsh.model.occ.importShapes(str(step_path))
			gmsh.model.occ.synchronize()
			self.model_dimensions = self._calculate_model_dimensions()
			progress_dialog.setLabelText("Finding exterior surfaces...")
			progress_dialog.setValue(40)
			QApplication.processEvents()

			volume_entities = gmsh.model.getEntities(3)
			if not volume_entities:
				raise ValueError("No valid 3D volume was found in the STEP file.")
			external_surface_ids = get_external_surface_ids(volume_entities)
			self.all_surfaces = set(external_surface_ids)

			# 외부 Surface의 2D mesh만 생성합니다. Volume mesh는 만들지 않습니다.
			gmsh.option.setNumber("Mesh.ElementOrder", 1)
			progress_dialog.setLabelText("Creating 2D surface mesh...")
			progress_dialog.setValue(55)
			QApplication.processEvents()
			gmsh.model.mesh.generate(2)
			points, node_index_by_tag = get_gmsh_nodes()

			progress_dialog.setLabelText("Building Viewer surfaces...")
			progress_dialog.setValue(75)
			QApplication.processEvents()
			for surface_id in external_surface_ids:
				polydata = surface_to_polydata(surface_id, points, node_index_by_tag)
				if polydata is None or polydata.n_cells == 0:
					continue

				actor = self.viewer.add_mesh(
					polydata,
					color="lightgray",
					show_edges=True,
					edge_color="dimgray",
					line_width=1.0,
					pickable=True,
					name=f"surface_{surface_id}",
				)
				self.actor_by_surface_id[surface_id] = actor
				self.surface_id_by_actor_key[actor_key(actor)] = surface_id

			# 파일을 다시 열었을 때도 새 actor를 클릭할 수 있도록
			# picking callback을 새 Viewer 내용에 연결합니다.
			self.viewer.enable_mesh_picking(
				callback=self.on_surface_picked,
				# PyVista 기본 분홍색 선택 표시를 끕니다.
				# 선택 색상은 _refresh_actor_colors()에서 직접 관리합니다.
				show=False,
				show_message=False,
				left_clicking=True,
				use_actor=True,
				picker="hardware",
			)
			self._fit_model_to_view()
			progress_dialog.setLabelText("Finished")
			progress_dialog.setValue(100)
			QApplication.processEvents()
			self.current_step_path = Path(step_path)
			self._update_target_mesh_size()
			self._invalidate_mesh()
			self.generate_mesh_button.setEnabled(True)
			self._sync_analysis_toolbar_buttons()
			self.setWindowTitle(
				f"Lead-Acid Battery Grid Resistance Analyzer - {self.current_step_path.name}"
			)
			print(f"Loaded STEP: {self.current_step_path}")
			print(f"Detected exterior surfaces: {len(self.all_surfaces)}")
			print("\nModel dimensions:")
			print(f"X: {self.model_dimensions['x']:.3f} mm")
			print(f"Y: {self.model_dimensions['y']:.3f} mm")
			print(f"Z: {self.model_dimensions['z']:.3f} mm")
			print("\nDetected:")
			print(f"Width: {self.model_dimensions['width']:.3f} mm")
			print(f"Height: {self.model_dimensions['height']:.3f} mm")
			print(f"Thickness: {self.model_dimensions['thickness']:.3f} mm")
			self._update_status()
		except Exception as error:
			self._clear_model_data()
			QMessageBox.critical(self, "STEP Load Error", str(error))
		finally:
			progress_dialog.close()

	def _clear_model_data(self):
		"""Viewer actor와 선택 집합, Gmsh 모델을 모두 비웁니다."""
		# 파일을 다시 열 때 기존 picking observer가 남지 않도록 제거합니다.
		# pyvistaqt 0.12.0에서는 disable_mesh_picking() 대신
		# 공통 함수인 disable_picking()을 사용합니다.
		self.viewer.disable_picking()
		self.viewer.clear()
		self.input_surfaces.clear()
		self.output_surfaces.clear()
		self.all_surfaces.clear()
		self.actor_by_surface_id.clear()
		self.surface_id_by_actor_key.clear()
		self.current_step_path = None
		self.model_dimensions = None
		self.analysis_config = None
		self._update_target_mesh_size()
		self.mesh_status_label.setText("Mesh Status: Not generated")
		self.generate_mesh_button.setEnabled(False)
		self._sync_analysis_toolbar_buttons()
		self.setWindowTitle("Lead-Acid Battery Grid Resistance Analyzer")
		if gmsh.isInitialized():
			gmsh.clear()
		self._set_empty_status()

	def _set_empty_status(self):
		"""STEP가 없는 빈 Viewer 상태를 표시합니다."""
		self.model_info_label.setText("File: No STEP file loaded")
		self.status_label.setText("No STEP file loaded")
		self.panel_boundary_status.setText(
			"Input Surfaces: 0\nOutput Surfaces: No model loaded"
		)

	def _on_material_changed(self, material_name):
		"""Library에서 선택한 Material의 두 reference 값을 함께 불러옵니다."""
		self.material_name = material_name
		material = next(
			(material for material in self.materials if material["name"] == material_name),
			None,
		)
		if material is not None:
			self.resistivity_edit.setText(f"{material['resistivity']:.2e}")
			self.reference_temperature_edit.setText(
				f"{material['reference_temperature']:.1f}"
			)
		self._update_conductivity()
		self._invalidate_mesh()

	def _open_fem_results(self):
		"""계산 결과를 별도 창에서 보여주고 기존 창은 앞으로 가져옵니다."""
		if self.fem_result is None:
			QMessageBox.warning(self, "FEM Results", "먼저 FEM Analysis를 실행해 주세요.")
			return
		try:
			if self.fem_results_dialog is None:
				self.fem_results_dialog = FemResultsDialog(
					self,
					self.fem_result,
					self.boundary_mode,
					self.model_dimensions,
					self.grid_real_battery_result,
				)
			self.fem_results_dialog.show()
			self.fem_results_dialog.raise_()
			self.fem_results_dialog.activateWindow()
		except Exception as error:
			QMessageBox.critical(self, "FEM Results Error", str(error))

	def _copy_fem_results(self):
		"""현재 FEM 분석 결과를 읽기 쉬운 텍스트로 시스템 클립보드에 복사합니다."""
		if self.fem_result is None:
			QMessageBox.warning(self, "Copy Results", "먼저 FEM Analysis를 실행해 주세요.")
			return
		result = self.fem_result
		hotspot = result["actual_hotspot"]
		centroid = hotspot["centroid_mm"]
		copy_text = "\n".join(
			[
				"=== Electrical FEM Analysis Result ===",
				f"Material: {self.material_combo.currentText()}",
				f"STEP: {self.current_step_path.name if self.current_step_path else 'N/A'}",
				"",
				"Operating Condition",
				f"Analysis Current: {result['analysis_current_a']:.6g} A",
				f"Reference Potential: {result['reference_potential_v']:.6g} V",
				f"Scale Factor: {result['scale_factor']:.6e}",
				"",
				"Electrical Results",
				f"Equivalent Resistance: {result['resistance_ohm']:.6e} Ohm",
				f"Equivalent Resistance: {result['resistance_mohm']:.6e} mOhm",
				f"Equivalent Resistance: {result['resistance_uohm']:.6e} microOhm",
				f"Conductance: {result['conductance_s']:.6e} S",
				f"Normalized Current @ 1 V: {result['current_a']:.6e} A",
				f"Voltage Drop: {result['voltage_drop_v']:.6e} V ({result['voltage_drop_v'] * 1000.0:.6e} mV)",
				f"Input Potential: {result['absolute_potential_max_v']:.6f} V",
				f"Output Potential: {result['absolute_potential_min_v']:.6f} V",
				f"Power Loss: {result['power_loss_w']:.6e} W",
				"",
				"FEM Results",
				f"Normalized Potential Min: {result['min_potential_v']:.6g} V",
				f"Normalized Potential Max: {result['max_potential_v']:.6g} V",
				f"Maximum Current Density: {np.max(result['actual_current_density_magnitude']):.6e} A/m^2",
				f"Maximum Current Density: {np.max(result['actual_current_density_magnitude_a_mm2']):.6e} A/mm^2",
				f"Maximum Joule Loss Density: {np.max(result['actual_joule_loss_density']):.6e} W/m^3",
				f"Integrated Joule Power: {result['actual_integrated_joule_power_w']:.6e} W",
				f"Power Difference: {result['actual_power_difference_percent']:.6g} %",
				"",
				"Hotspot",
				f"Element Index: {hotspot['element_index']}",
				f"Joule Loss Density: {hotspot['joule_loss_density_w_m3']:.6e} W/m^3",
				f"Current Density: {hotspot['current_density_a_m2']:.6e} A/m^2",
				f"Current Density: {hotspot['current_density_a_mm2']:.6e} A/mm^2",
				f"Position: X={centroid[0]:.6g} mm, Y={centroid[1]:.6g} mm, Z={centroid[2]:.6g} mm",
			]
		)
		QApplication.clipboard().setText(copy_text)
		QMessageBox.information(self, "Copy Results", "분석 결과를 클립보드에 복사했습니다.")

	def _apply_scaled_fem_result(self):
		"""저장된 1 V 결과를 현재 전류 조건으로 다시 계산합니다."""
		if self.normalized_fem_result is None:
			return
		was_results_visible = (
			self.fem_results_dialog is not None and self.fem_results_dialog.isVisible()
		)
		self.fem_result = dict(
			scale_fem_result_to_current(
				self.normalized_fem_result,
				self.analysis_current_spin.value(),
				self.reference_potential_spin.value(),
			)
		)
		self.fem_result.update(
			{
				key: self.normalized_fem_result[key]
				for key in self.normalized_fem_result
				if key not in self.fem_result
			}
		)
		result = self.fem_result
		self.fem_voltage_drop_label.setText(
			f"{result['voltage_drop_v']:.6g} V ({result['voltage_drop_v'] * 1000.0:.6g} mV)"
		)
		self.fem_power_loss_label.setText(f"{result['power_loss_w']:.6e}")
		self.fem_input_potential_label.setText(f"{result['absolute_potential_max_v']:.6f}")
		self.fem_output_potential_label.setText(f"{result['absolute_potential_min_v']:.6f}")
		self.fem_max_current_density_label.setText(
			f"{np.max(result['actual_current_density_magnitude_a_mm2']):.6e}"
		)
		self.fem_max_joule_loss_label.setText(
			f"{np.max(result['actual_joule_loss_density']):.6e}"
		)
		if self.fem_results_dialog is not None:
			self.fem_results_dialog.close()
			self.fem_results_dialog = None
		if was_results_visible:
			self._open_fem_results()

	def _on_analysis_current_changed(self):
		"""Analysis Current 변경은 FEM 재계산 없이 scaled 결과만 갱신합니다."""
		if self.normalized_fem_result is not None:
			try:
				if self.grid_real_battery_result is not None:
					self.grid_real_battery_result = scale_real_battery_result_to_current(
						self.grid_real_battery_result, self.analysis_current_spin.value()
					)
				self._apply_scaled_fem_result()
			except Exception as error:
				QMessageBox.warning(self, "Analysis Current", str(error))

	def _invalidate_real_battery_result(self):
		"""Gaussian spread 변경은 Characterization 결과를 건드리지 않습니다."""
		self.grid_real_battery_result = None
		if self.fem_results_dialog is not None:
			self.fem_results_dialog.close()
			self.fem_results_dialog = None
		if self.boundary_mode == 1 and self.grid_characterization_result is not None:
			self.fem_status_label.setText("FEM Status: Real Battery Model needs recalculation")

	def _on_reference_potential_changed(self):
		"""Reference Potential 변경 시 solver 없이 절대전위 offset만 갱신합니다."""
		if self.normalized_fem_result is not None:
			try:
				self._apply_scaled_fem_result()
			except Exception as error:
				QMessageBox.warning(self, "Reference Potential", str(error))

	def run_fem_analysis(self):
		"""현재 Mesh, 경계, Material로 FEM solver를 실행하고 결과를 표시합니다."""
		if self.current_step_path is None:
			QMessageBox.warning(self, "FEM Analysis", "먼저 STEP 파일을 열어 주세요.")
			return
		if self.analysis_config is None or self.mesh_status_label.text().split("\n")[0] != "Mesh Status: Ready":
			QMessageBox.warning(self, "FEM Analysis", "먼저 유효한 3D Mesh를 생성해 주세요.")
			return
		material = self.get_material_properties()
		if material["conductivity"] is None or material["conductivity"] <= 0:
			QMessageBox.warning(self, "FEM Analysis", "Electrical Conductivity가 유효하지 않습니다.")
			return
		physical_groups = self.analysis_config.get("physical_groups", {})
		if not physical_groups.get("input") or not physical_groups.get("output"):
			QMessageBox.warning(self, "FEM Analysis", "Input/Output Physical Group이 없습니다.")
			return

		progress_dialog = QProgressDialog("Preparing FEM mesh...", None, 0, 5, self)
		progress_dialog.setWindowTitle("FEM Analysis")
		progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
		progress_dialog.setMinimumDuration(0)
		progress_dialog.setValue(0)
		self.fem_status_label.setText("FEM Status: Solving...")
		self.run_fem_button.setEnabled(False)
		self._sync_analysis_toolbar_buttons()
		QApplication.processEvents()
		try:
			progress_dialog.setLabelText("Mapping boundaries...")
			progress_dialog.setValue(1)
			QApplication.processEvents()
			progress_dialog.setLabelText("Assembling conduction matrix...")
			progress_dialog.setValue(2)
			QApplication.processEvents()
			progress_dialog.setLabelText("Solving potential...")
			progress_dialog.setValue(3)
			QApplication.processEvents()
			result = run_electrical_fem(
				self.analysis_config["input_surfaces"],
				self.analysis_config["output_surfaces"],
				material["conductivity"],
			)
			self.grid_characterization_result = result if self.boundary_mode == 1 else None
			if self.boundary_mode == 1:
				progress_dialog.setLabelText("Running Real Battery Gaussian Current Model...")
				QApplication.processEvents()
				model_box = gmsh.model.getBoundingBox(-1, -1)
				gaussian_center = ((model_box[0] + model_box[3]) / 2.0, (model_box[1] + model_box[4]) / 2.0)
				self.grid_real_battery_result = run_real_battery_gaussian_fem(
					self.analysis_config["input_surfaces"],
					set(self.all_surfaces) - set(self.analysis_config["input_surfaces"]),
					material["conductivity"],
					self.analysis_current_spin.value(),
					gaussian_center,
					(self.gaussian_spread_x_spin.value(), self.gaussian_spread_y_spin.value()),
				)
			progress_dialog.setLabelText("Calculating resistance...")
			progress_dialog.setValue(4)
			QApplication.processEvents()
			self.normalized_fem_result = result
			self._apply_scaled_fem_result()
			self.fem_min_potential_label.setText(f"{result['min_potential_v']:.6g}")
			self.fem_max_potential_label.setText(f"{result['max_potential_v']:.6g}")
			self.fem_resistance_label.setText(f"{result['resistance_uohm']:.6g}")
			self.fem_conductance_label.setText(f"{result['conductance_s']:.6e}")
			self.fem_current_label.setText(f"{result['current_a']:.6e}")
			self.fem_status_label.setText("FEM Status: Ready")
			self.view_fem_results_button.setEnabled(True)
			self.copy_fem_results_button.setEnabled(True)
			self._sync_analysis_toolbar_buttons()
			result = self.fem_result
			progress_dialog.setValue(5)
			progress_dialog.setLabelText("Finished")
			print("\n=== Electrical FEM Result ===")
			print(f"Material: {material['material']}")
			print(f"Resistivity: {material['resistivity']:.6e} Ohm m")
			print(f"Conductivity: {material['conductivity']:.6e} S/m")
			print(f"Mesh Nodes: {result['mesh_nodes']}")
			print(f"Tetrahedra: {result['tetrahedra']}")
			print(f"DOFs: {result['dofs']}")
			print(f"Input FEM facets: {result['input_facets']}")
			print(f"Output FEM facets: {result['output_facets']}")
			print(f"Potential Min/Max: {result['min_potential_v']:.6g} / {result['max_potential_v']:.6g} V")
			print(f"Conductance: {result['conductance_s']:.6e} S")
			print(f"Total Current: {result['current_a']:.6e} A")
			print(f"Resistance: {result['resistance_ohm']:.6e} Ohm")
			print(f"Resistance: {result['resistance_mohm']:.6e} mOhm")
			print(f"Resistance: {result['resistance_uohm']:.6e} microOhm")
			print("\n=== FEM Post Processing ===")
			print(f"Maximum Current Density: {np.max(result['actual_current_density_magnitude']):.6e} A/m^2")
			print(f"Maximum Current Density: {np.max(result['actual_current_density_magnitude_a_mm2']):.6e} A/mm^2")
			print(f"Maximum Joule Loss Density: {np.max(result['actual_joule_loss_density']):.6e} W/m^3")
			print(f"Energy Power: {result['power_loss_w']:.6e} W")
			print(f"Integrated Actual Joule Power: {result['actual_integrated_joule_power_w']:.6e} W")
			print(f"Power Difference: {result['actual_power_difference_percent']:.6g} %")
			print(f"Analysis Current: {result['analysis_current_a']:.6g} A")
			print(f"Reference Potential: {result['reference_potential_v']:.6g} V")
			print(f"Scale Factor: {result['scale_factor']:.6e}")
			print(f"Voltage Drop: {result['voltage_drop_v'] * 1000.0:.6g} mV")
			print(f"Input Absolute Potential: {result['absolute_potential_max_v']:.6f} V")
			print(f"Output Absolute Potential: {result['absolute_potential_min_v']:.6f} V")
			print(f"Power Loss: {result['power_loss_w']:.6e} W")
			print(f"Hotspot Element: {result['actual_hotspot']['element_index']}")
			centroid = result['actual_hotspot']['centroid_mm']
			print(f"Hotspot Position: X={centroid[0]:.6g} mm, Y={centroid[1]:.6g} mm, Z={centroid[2]:.6g} mm")
			if result["flux_current_a"] is None:
				print("Flux verification: unavailable")
			else:
				print(f"Input flux current: {result['flux_current_a']:.6e} A")
				print(f"Flux difference: {result['flux_difference_percent']:.6g} %")
		except Exception as error:
			self.fem_result = None
			self.normalized_fem_result = None
			self.fem_status_label.setText("FEM Status: Failed")
			self.view_fem_results_button.setEnabled(False)
			self.copy_fem_results_button.setEnabled(False)
			QMessageBox.critical(self, "FEM Analysis Error", str(error))
		finally:
			if self.analysis_config is not None:
				self.run_fem_button.setEnabled(True)
			self._sync_analysis_toolbar_buttons()
			progress_dialog.close()

	def _validate_material_record(self, material, editing_index=None):
		"""Add/Edit에서 입력한 Material record를 검증하고 오류를 안내합니다."""
		if not material["name"]:
			QMessageBox.warning(self, "Invalid Material", "Material Name은 비어 있을 수 없습니다.")
			return False
		try:
			if material["resistivity"] <= 0:
				raise ValueError
		except (TypeError, ValueError):
			QMessageBox.warning(self, "Invalid Material", "Resistivity는 0보다 큰 숫자여야 합니다.")
			return False
		for index, existing in enumerate(self.materials):
			if index != editing_index and existing["name"].casefold() == material["name"].casefold():
				QMessageBox.warning(self, "Duplicate Material", "동일한 Material 이름이 이미 존재합니다.")
				return False
		return True

	def _open_material_manager(self):
		"""Material Library 창을 열고 Add/Edit/Delete 결과를 즉시 반영합니다."""
		library_dialog = MaterialLibraryDialog(self, self.materials)
		library_dialog.refresh(self.material_combo.currentText())
		library_dialog.add_button.clicked.connect(
			lambda: self._add_material(library_dialog)
		)
		library_dialog.edit_button.clicked.connect(
			lambda: self._edit_material(library_dialog)
		)
		library_dialog.delete_button.clicked.connect(
			lambda: self._delete_material(library_dialog)
		)
		library_dialog.exec()

	def _add_material(self, library_dialog):
		"""새 Material을 검증한 뒤 Library와 메인 ComboBox에 추가합니다."""
		edit_dialog = MaterialEditDialog(self)
		if edit_dialog.exec() != QDialog.DialogCode.Accepted:
			return
		try:
			material = edit_dialog.get_material()
		except ValueError:
			QMessageBox.warning(self, "Invalid Material", "모든 값은 숫자로 입력해야 합니다.")
			return
		if not self._validate_material_record(material):
			return
		self.materials.append(material)
		self._save_material_library()
		self._refresh_material_combo(material["name"])
		library_dialog.refresh(material["name"])

	def _edit_material(self, library_dialog):
		"""선택된 Material을 편집하고 기존 record를 교체합니다."""
		index = library_dialog.list_widget.currentRow()
		if index < 0:
			return
		old_name = self.materials[index]["name"]
		edit_dialog = MaterialEditDialog(self, self.materials[index])
		if edit_dialog.exec() != QDialog.DialogCode.Accepted:
			return
		try:
			material = edit_dialog.get_material()
		except ValueError:
			QMessageBox.warning(self, "Invalid Material", "모든 값은 숫자로 입력해야 합니다.")
			return
		if not self._validate_material_record(material, index):
			return
		self.materials[index] = material
		self._save_material_library()
		selected_name = material["name"] if self.material_combo.currentText() == old_name else self.material_combo.currentText()
		self._refresh_material_combo(selected_name)
		library_dialog.refresh(material["name"])

	def _delete_material(self, library_dialog):
		"""삭제 전 확인하고, 마지막 Material은 삭제하지 않습니다."""
		if len(self.materials) <= 1:
			QMessageBox.information(
				self, "Material Library", "At least one material must remain in the library."
			)
			return
		index = library_dialog.list_widget.currentRow()
		if index < 0:
			return
		name = self.materials[index]["name"]
		reply = QMessageBox.question(
			self,
			"Delete Material",
			f'Delete material "{name}"?',
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
		)
		if reply != QMessageBox.StandardButton.Yes:
			return
		was_selected = self.material_combo.currentText() == name
		self.materials.pop(index)
		self._save_material_library()
		self._refresh_material_combo(None if was_selected else self.material_combo.currentText())
		library_dialog.refresh(self.material_combo.currentText())

	def _validate_resistivity(self):
		"""비저항이 0보다 큰 유효한 숫자인지 시각적으로 표시합니다."""
		try:
			is_valid = float(self.resistivity_edit.text()) > 0
		except ValueError:
			is_valid = False
		if not is_valid:
			self.resistivity_edit.setStyleSheet("QLineEdit { border: 1px solid #c0392b; }")
			self.conductivity_edit.setText("Invalid resistivity")
		else:
			self.resistivity_edit.setStyleSheet("")
			self._update_conductivity()

	def _update_conductivity(self):
		"""비저항 입력이 바뀔 때 전기전도도를 즉시 다시 계산합니다."""
		try:
			resistivity = float(self.resistivity_edit.text())
			if resistivity <= 0:
				raise ValueError
			self.conductivity_edit.setText(f"{1.0 / resistivity:.6e}")
			self.resistivity_edit.setStyleSheet("")
		except ValueError:
			self.conductivity_edit.setText("Invalid resistivity")
			self.resistivity_edit.setStyleSheet(
				"QLineEdit { border: 1px solid #c0392b; }"
			)

	def get_material_properties(self):
		"""현재 Material 값을 향후 해석 코드가 읽을 수 있는 형태로 반환합니다."""
		try:
			resistivity = float(self.resistivity_edit.text())
			reference_temperature = float(self.reference_temperature_edit.text())
			if resistivity <= 0:
				raise ValueError
			conductivity = 1.0 / resistivity
		except ValueError:
			return {
				"material": self.material_combo.currentText(),
				"resistivity": None,
				"conductivity": None,
				"reference_temperature": None,
			}
		return {
			"material": self.material_combo.currentText(),
			"resistivity": resistivity,
			"conductivity": conductivity,
			"reference_temperature": reference_temperature,
		}

	def _calculate_model_dimensions(self):
		"""동기화된 전체 3D 모델의 치수와 방향별 역할을 계산합니다."""
		# -1, -1은 특정 Entity가 아닌 전체 모델의 Bounding Box를 뜻합니다.
		min_x, min_y, min_z, max_x, max_y, max_z = gmsh.model.getBoundingBox(-1, -1)
		dimensions_by_axis = {
			"x": max_x - min_x,
			"y": max_y - min_y,
			"z": max_z - min_z,
		}

		# 세 방향의 길이를 큰 순서로 정렬하면 큰 두 값 중 큰 값은 Width,
		# 작은 값은 Height가 되고, 가장 작은 값은 Thickness 후보가 됩니다.
		sorted_axes = [
			axis
			for axis, _ in sorted(
				dimensions_by_axis.items(),
				key=lambda item: item[1],
				reverse=True,
			)
		]
		ordered_dimensions = [dimensions_by_axis[axis] for axis in sorted_axes]
		return {
			**dimensions_by_axis,
			"width": ordered_dimensions[0],
			"height": ordered_dimensions[1],
			"thickness": ordered_dimensions[2],
			"width_axis": sorted_axes[0],
			"height_axis": sorted_axes[1],
			"thickness_axis": sorted_axes[2],
		}

	def _fit_model_to_view(self):
		"""모델 크기에 맞춰 카메라를 초기화하고 보기 좋은 크기로 확대합니다."""
		# reset_camera()는 현재 모델의 Bounding Box를 기준으로 화면을 맞춥니다.
		# 모델의 실제 크기를 하드코딩하지 않고, 화면에서 조금 더 크게 보이도록 합니다.
		self.viewer.reset_camera()
		self.viewer.camera.zoom(1.2)
		self.viewer.render()

	def open_step_file(self):
		"""File 메뉴에서 STEP 파일을 선택해 엽니다."""
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Open STEP File",
			str(Path(__file__).resolve().parent),
			"STEP Files (*.step *.stp)",
		)
		if file_path:
			self._load_step_and_build_viewer(Path(file_path))

	def close_model(self):
		"""현재 STEP 모델을 닫고 빈 Viewer로 되돌립니다."""
		self._clear_model_data()
		self._update_controls()
		print("Model closed.")

	def reset_camera(self):
		"""현재 Viewer의 모델 전체가 보이도록 카메라를 초기화합니다."""
		self._fit_model_to_view()

	def set_isometric_view(self):
		"""모델을 대각선 방향에서 보는 등각 투영으로 전환합니다."""
		self.viewer.view_isometric()
		self.viewer.render()

	def _set_axis_view(self, axis, negative=False):
		"""지정한 축의 양 또는 음 방향을 바라보는 표준 View를 설정합니다."""
		view_by_axis = {
			"x": self.viewer.view_yz,
			"y": self.viewer.view_xz,
			"z": self.viewer.view_xy,
		}
		view_by_axis[axis](negative=negative)
		self.viewer.render()

	def set_top_view(self):
		"""가장 얇은 방향을 바라보는 Grid 평면 View로 전환합니다."""
		if self.model_dimensions is None:
			return
		self._set_axis_view(self.model_dimensions["thickness_axis"])

	def set_bottom_view(self):
		"""Top View의 반대 방향인 Bottom View로 전환합니다."""
		if self.model_dimensions is None:
			return
		self._set_axis_view(self.model_dimensions["thickness_axis"], negative=True)

	def set_front_view(self):
		"""자동 판정된 Height 축 방향에서 보는 Front View로 전환합니다."""
		if self.model_dimensions is None:
			return
		self._set_axis_view(self.model_dimensions["height_axis"])

	def set_back_view(self):
		"""Front View의 반대 방향인 Back View로 전환합니다."""
		if self.model_dimensions is None:
			return
		self._set_axis_view(self.model_dimensions["height_axis"], negative=True)

	def set_right_view(self):
		"""Width 축의 정방향에서 보는 Right View로 전환합니다."""
		if self.model_dimensions is None:
			return
		self._set_axis_view(self.model_dimensions["width_axis"])

	def set_left_view(self):
		"""Right View의 반대 방향인 Left View로 전환합니다."""
		if self.model_dimensions is None:
			return
		self._set_axis_view(self.model_dimensions["width_axis"], negative=True)

	def show_about(self):
		"""프로그램 정보를 표시합니다."""
		QMessageBox.about(
			self,
			"About",
			"Lead-Acid Battery Grid Resistance Analyzer\n\n"
			"Seo, Seyoung\n"
			"ssy523a@gamil.com",
		)

	def _refresh_actor_colors(self):
		"""현재 Input/Output 집합에 따라 모든 actor 색상을 갱신합니다."""
		for surface_id, actor in self.actor_by_surface_id.items():
			if surface_id in self.input_surfaces:
				actor.GetProperty().SetColor(0.9, 0.1, 0.1)
			elif surface_id in self.output_surfaces:
				actor.GetProperty().SetColor(0.1, 0.3, 0.9)
			else:
				actor.GetProperty().SetColor(0.83, 0.83, 0.83)
		self.viewer.render()

	def _update_status(self):
		"""현재 모드와 Input/Output/Insulated 개수를 Label에 표시합니다."""
		if self.current_step_path is None:
			self._set_empty_status()
			return
		self._update_model_info()
		mode_text = f"Mode: Option {self.boundary_mode}"
		target_text = f"Selection target: {'Input' if self.selection_type == 'input' else 'Output'}"
		input_text = f"Input Surface: {len(self.input_surfaces)}"
		if self.boundary_mode == 1:
			output_text = "Output Surface: Automatic"
			insulated_text = ""
		else:
			output_text = f"Output Surface: {len(self.output_surfaces)}"
			insulated_count = len(
				self.all_surfaces - self.input_surfaces - self.output_surfaces
			)
			insulated_text = f"Insulated Surface: {insulated_count}"
		status_text = " | ".join(
			filter(None, [mode_text, target_text, input_text, output_text, insulated_text])
		)
		self.status_label.setText(status_text)
		self.panel_boundary_status.setText(
			f"Input Surfaces: {len(self.input_surfaces)}\n"
			f"{output_text}"
			+ (f"\n{insulated_text}" if insulated_text else "")
		)

	def _update_model_info(self):
		"""현재 파일명과 자동 판정된 Grid 치수를 한 줄로 표시합니다."""
		if self.current_step_path is None or self.model_dimensions is None:
			self.model_info_label.setText("File: No STEP file loaded")
			return
		dimensions = self.model_dimensions
		self.model_info_label.setText(
			f"File: {self.current_step_path.name} | "
			f"Width: {dimensions['width']:.3f} mm | "
			f"Height: {dimensions['height']:.3f} mm | "
			f"Thickness: {dimensions['thickness']:.3f} mm"
		)

	def _update_controls(self):
		"""모드에 맞게 버튼 활성화와 선택 강조 상태를 갱신합니다."""
		self.option_1_action.setChecked(self.boundary_mode == 1)
		self.option_2_action.setChecked(self.boundary_mode == 2)
		self.output_button.setEnabled(self.boundary_mode == 2)
		self.input_button.setChecked(self.selection_type == "input")
		self.output_button.setChecked(
			self.boundary_mode == 2 and self.selection_type == "output"
		)
		# Toolbar와 좌측 Control Panel이 같은 내부 상태를 표시하도록 합니다.
		self.panel_option_1.setChecked(self.boundary_mode == 1)
		self.panel_option_2.setChecked(self.boundary_mode == 2)
		self.panel_output_button.setEnabled(self.boundary_mode == 2)
		self.panel_input_button.setChecked(self.selection_type == "input")
		self.panel_output_button.setChecked(
			self.boundary_mode == 2 and self.selection_type == "output"
		)
		self._update_status()

	def set_boundary_mode(self, mode):
		"""Option 1 또는 Option 2로 전환합니다."""
		self.boundary_mode = mode
		# 모드 전환 시 이전에 계산되거나 직접 선택된 Output을 섞지 않습니다.
		self.output_surfaces.clear()
		self._invalidate_mesh()
		self._refresh_actor_colors()
		self._update_controls()
		print(f"Current mode: Option {mode}")

	def set_selection_type(self, selection_type):
		"""이후 클릭할 Surface의 대상 집합을 바꿉니다."""
		if self.boundary_mode == 1 and selection_type == "output":
			return
		self.selection_type = selection_type
		self._invalidate_mesh()
		self._update_controls()
		print(f"Current selection target: {'Input' if selection_type == 'input' else 'Output'}")

	def on_surface_picked(self, picked_actor):
		"""클릭된 actor의 Gmsh Surface ID를 현재 집합에 토글합니다."""
		if picked_actor is None:
			return
		surface_id = self.surface_id_by_actor_key.get(actor_key(picked_actor))
		if surface_id is None:
			return

		if self.boundary_mode == 1 or self.selection_type == "input":
			if surface_id in self.input_surfaces:
				self.input_surfaces.remove(surface_id)
			else:
				self.output_surfaces.discard(surface_id)
				self.input_surfaces.add(surface_id)
		else:
			if surface_id in self.output_surfaces:
				self.output_surfaces.remove(surface_id)
			else:
				self.input_surfaces.discard(surface_id)
				self.output_surfaces.add(surface_id)

		self._refresh_actor_colors()
		self._invalidate_mesh()
		self._update_status()
		print(f"Selected Surface ID: {surface_id}")
		print(f"Input Surface ID: {sorted(self.input_surfaces)}")
		print(f"Output Surface ID: {sorted(self.output_surfaces)}")

	def confirm_selection(self):
		"""현재 모드에 맞춰 Output 또는 Insulated 집합을 계산하고 출력합니다."""
		if self.boundary_mode == 1:
			self.output_surfaces = self.all_surfaces - self.input_surfaces
			print("\n--- Option 1 Confirmed ---")
			print(f"Total exterior surfaces: {len(self.all_surfaces)}")
			print(f"Input surfaces: {len(self.input_surfaces)}")
			print(f"Output surfaces: {len(self.output_surfaces)}")
			print(f"Input Surface ID: {sorted(self.input_surfaces)}")
		else:
			insulated_surfaces = (
				self.all_surfaces - self.input_surfaces - self.output_surfaces
			)
			print("\n--- Option 2 Confirmed ---")
			print(f"Total exterior surfaces: {len(self.all_surfaces)}")
			print(f"Input surfaces: {len(self.input_surfaces)}")
			print(f"Output surfaces: {len(self.output_surfaces)}")
			print(f"Insulated surfaces: {len(insulated_surfaces)}")
			print(f"Input Surface IDs: {sorted(self.input_surfaces)}")
			print(f"Output Surface IDs: {sorted(self.output_surfaces)}")
		self._refresh_actor_colors()
		self._invalidate_mesh()
		self._update_status()

	def clear_selection(self):
		"""Input과 Output을 비우고 모든 actor를 회색으로 되돌립니다."""
		self.input_surfaces.clear()
		self.output_surfaces.clear()
		self._invalidate_mesh()
		self._refresh_actor_colors()
		self._update_status()
		print("\nAll Input and Output surface selections have been cleared.")

	def closeEvent(self, event):
		"""창을 닫을 때 PyVista와 Gmsh 리소스를 정리합니다."""
		if self.mesh_preview_dialog is not None:
			self.mesh_preview_dialog.close()
			self.mesh_preview_dialog = None
		if self.fem_results_dialog is not None:
			self.fem_results_dialog.close()
			self.fem_results_dialog = None
		self.viewer.close()
		if gmsh.isInitialized():
			gmsh.finalize()
		event.accept()


def main():
	"""Qt 애플리케이션을 시작합니다."""
	app = QApplication.instance() or QApplication(sys.argv)
	app.setOrganizationName("Lead-Acid Battery")
	app.setApplicationName("Grid Resistance Analyzer")
	if APP_ICON_PATH.exists():
		app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
	window = SurfaceSelectionWindow()
	window.show()
	return app.exec()


if __name__ == "__main__":
	sys.exit(main())
