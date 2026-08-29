"""STEP 외부 Surface를 PyVista에서 마우스로 선택하는 테스트입니다."""

from collections import Counter
from pathlib import Path

import gmsh
import numpy as np
import pyvista as pv


def get_external_surface_ids(volume_entities):
	"""한 개의 Volume에만 속한 외부 Surface ID를 반환합니다."""
	surface_occurrences = Counter()
	for volume_entity in volume_entities:
		# Volume 바로 아래의 Surface만 조사합니다.
		for dimension, surface_id in gmsh.model.getBoundary(
			[volume_entity], oriented=False, recursive=False
		):
			if dimension == 2:
				surface_occurrences[surface_id] += 1

	return sorted(
		surface_id
		for surface_id, occurrence_count in surface_occurrences.items()
		if occurrence_count == 1
	)


def get_gmsh_nodes():
	"""Gmsh 전체 2D mesh node를 PyVista가 사용할 좌표 배열로 변환합니다."""
	# Surface마다 node를 따로 요청하면 같은 node의 좌표가 반복될 수 있으므로
	# 전체 node를 한 번만 읽고, Gmsh node tag를 배열 위치로 연결합니다.
	node_tags, coordinates, _ = gmsh.model.mesh.getNodes(
		-1, -1, includeBoundary=True, returnParametricCoord=False
	)
	points = np.asarray(coordinates, dtype=float).reshape((-1, 3))
	node_index_by_tag = {
		int(node_tag): index for index, node_tag in enumerate(node_tags)
	}
	return points, node_index_by_tag


def surface_to_polydata(surface_id, points, node_index_by_tag):
	"""Gmsh의 한 Surface 2D mesh를 PyVista PolyData로 변환합니다."""
	element_types, _, element_node_tags = gmsh.model.mesh.getElements(2, surface_id)
	faces = []

	for element_type, node_tags in zip(element_types, element_node_tags):
		# 이번 테스트에서는 1차 삼각형/사각형만 사용합니다.
		# 아래 Mesh.ElementOrder 설정으로 일반적으로 삼각형이 생성됩니다.
		_, _, _, number_of_nodes, _, _ = gmsh.model.mesh.getElementProperties(
			element_type
		)
		if number_of_nodes not in (3, 4):
			continue

		elements = np.asarray(node_tags, dtype=np.int64).reshape((-1, number_of_nodes))
		for element in elements:
			try:
				point_indices = [node_index_by_tag[int(tag)] for tag in element]
			except KeyError:
				# 해당 Surface에 속한 node 좌표를 찾지 못하면 해당 요소만 건너뜁니다.
				continue
			faces.extend([number_of_nodes, *point_indices])

	if not faces:
		return None

	return pv.PolyData(points, np.asarray(faces, dtype=np.int64))


def actor_key(actor):
	"""PyVista callback에서 받은 VTK actor를 식별할 안정적인 키를 만듭니다."""
	return actor.GetAddressAsString("")


def main():
	# 실행 위치와 관계없이 이 Python 파일 옆의 grid.step을 읽습니다.
	step_path = Path(__file__).resolve().parent / "grid.step"
	# boundary_mode은 현재 경계 선택 방법을 나타냅니다.
	# 1: Input만 선택하고 나머지는 자동 Output
	# 2: Input과 Output을 각각 직접 선택
	boundary_mode = 1
	selection_type = "input"
	input_surfaces = set()
	output_surfaces = set()
	actor_by_surface_id = {}
	surface_id_by_actor_key = {}

	try:
		# Gmsh OpenCASCADE로 STEP 파일을 불러옵니다.
		gmsh.initialize()
		gmsh.option.setNumber("General.Terminal", 1)
		gmsh.model.add("grid_from_step")
		gmsh.model.occ.importShapes(str(step_path))
		gmsh.model.occ.synchronize()

		volume_entities = gmsh.model.getEntities(3)
		external_surface_ids = get_external_surface_ids(volume_entities)
		# 모든 외부 Surface를 set으로 보관합니다.
		# 나중에 Option 1의 Output은 이 집합에서 Input을 빼서 계산합니다.
		all_surfaces = set(external_surface_ids)
		print(f"인식된 3D Volume 수: {len(volume_entities)}")
		print(f"인식된 외부 Surface 수: {len(external_surface_ids)}")

		# PyVista에 표시할 2D Surface mesh만 생성합니다.
		# 3D Volume mesh는 생성하지 않습니다.
		gmsh.option.setNumber("Mesh.ElementOrder", 1)
		gmsh.model.mesh.generate(2)
		points, node_index_by_tag = get_gmsh_nodes()

		# Windows 창 제목의 한글 인코딩 문제를 피하기 위해 제목은 영문으로 표시합니다.
		plotter = pv.Plotter(title="STEP Surface Selection Test")
		plotter.set_background("white")

		# Surface별로 actor를 따로 만들면 클릭한 actor에서 Surface ID를 역조회할 수 있습니다.
		for surface_id in external_surface_ids:
			polydata = surface_to_polydata(surface_id, points, node_index_by_tag)
			if polydata is None or polydata.n_cells == 0:
				continue

			actor = plotter.add_mesh(
				polydata,
				color="lightgray",
				show_edges=False,
				pickable=True,
				name=f"surface_{surface_id}",
			)
			actor_by_surface_id[surface_id] = actor
			surface_id_by_actor_key[actor_key(actor)] = surface_id

		print(f"PyVista에 표시한 외부 Surface 수: {len(actor_by_surface_id)}")

		def on_surface_picked(picked_actor):
			"""마우스로 클릭된 actor의 Surface ID를 토글합니다."""
			if picked_actor is None:
				return

			surface_id = surface_id_by_actor_key.get(actor_key(picked_actor))
			if surface_id is None:
				print("Surface ID를 확인하지 못했습니다.")
				return

			if boundary_mode == 1:
				# Option 1에서는 클릭한 면을 Input으로만 관리합니다.
				# Output은 Enter를 누르는 순간 자동으로 계산합니다.
				if surface_id in input_surfaces:
					input_surfaces.remove(surface_id)
				else:
					input_surfaces.add(surface_id)
			else:
				# Option 2에서는 현재 선택 대상에 따라 색상과 집합을 결정합니다.
				# 한 Surface가 Input과 Output에 동시에 들어가지 않도록
				# 반대쪽 집합에서 먼저 제거합니다.
				if selection_type == "input":
					if surface_id in input_surfaces:
						input_surfaces.remove(surface_id)
					else:
						output_surfaces.discard(surface_id)
						input_surfaces.add(surface_id)
				else:
					if surface_id in output_surfaces:
						output_surfaces.remove(surface_id)
					else:
						input_surfaces.discard(surface_id)
						output_surfaces.add(surface_id)

			# 모든 actor의 색상을 현재 집합 상태에 맞춰 다시 지정합니다.
			# 회색은 아직 선택하지 않은 Surface의 기본 색상입니다.
			for actor_surface_id, actor in actor_by_surface_id.items():
				if actor_surface_id in input_surfaces:
					actor.GetProperty().SetColor(0.9, 0.1, 0.1)
				elif actor_surface_id in output_surfaces:
					actor.GetProperty().SetColor(0.1, 0.3, 0.9)
				else:
					actor.GetProperty().SetColor(0.83, 0.83, 0.83)

			plotter.render()
			print(f"\n선택된 Surface ID: {surface_id}")
			print(f"현재 Input Surface 수: {len(input_surfaces)}")
			print(f"현재 Output Surface 수: {len(output_surfaces)}")
			print(f"현재 Input Surface ID 목록: {sorted(input_surfaces)}")
			print(f"현재 Output Surface ID 목록: {sorted(output_surfaces)}")

		def show_boundary_mode():
			"""현재 Boundary Mode를 Terminal에 표시합니다."""
			print(f"\n현재 Boundary Mode: Option {boundary_mode}")
			if boundary_mode == 2:
				print(
					f"현재 선택 대상: {'Input' if selection_type == 'input' else 'Output'}"
				)

		def set_option_1():
			"""Boundary Mode를 Option 1로 변경합니다."""
			nonlocal boundary_mode, output_surfaces
			boundary_mode = 1
			# Option 1의 Output은 Enter를 누를 때마다 자동 계산하므로,
			# Option 2에서 직접 고른 Output이 남지 않도록 초기화합니다.
			output_surfaces.clear()
			for surface_id, actor in actor_by_surface_id.items():
				if surface_id in input_surfaces:
					actor.GetProperty().SetColor(0.9, 0.1, 0.1)
				else:
					actor.GetProperty().SetColor(0.83, 0.83, 0.83)
			plotter.render()
			show_boundary_mode()

		def set_option_2():
			"""Boundary Mode를 Option 2로 변경합니다."""
			nonlocal boundary_mode, output_surfaces
			boundary_mode = 2
			# Option 1에서 자동으로 계산된 Output은 Option 2의 직접 선택
			# Output이 아니므로 모드를 바꿀 때 비웁니다.
			output_surfaces.clear()
			show_boundary_mode()

		def select_input():
			"""Option 2에서 이후 클릭 대상을 Input으로 설정합니다."""
			nonlocal selection_type
			selection_type = "input"
			print("\n현재 선택 대상: Input")

		def select_output():
			"""Option 2에서 이후 클릭 대상을 Output으로 설정합니다."""
			nonlocal selection_type
			selection_type = "output"
			print("\n현재 선택 대상: Output")

		def confirm_option_1():
			"""현재 Input 선택을 확정하고 나머지를 Output으로 계산합니다."""
			nonlocal output_surfaces
			if boundary_mode == 1:
				# Option 1에서는 사용자가 고른 면만 Input이고, 나머지 외부 면은
				# 별도의 클릭 없이 자동으로 Output이 됩니다.
				output_surfaces = all_surfaces - input_surfaces
				# Option 1 확정 후 화면에서도 Input/Output을 구분합니다.
				# Input은 빨간색, 자동 Output은 파란색으로 표시합니다.
				for surface_id, actor in actor_by_surface_id.items():
					if surface_id in input_surfaces:
						actor.GetProperty().SetColor(0.9, 0.1, 0.1)
					elif surface_id in output_surfaces:
						actor.GetProperty().SetColor(0.1, 0.3, 0.9)
					else:
						actor.GetProperty().SetColor(0.83, 0.83, 0.83)
				plotter.render()
				print("\n--- Option 1 확정 ---")
				print(f"전체 외부 Surface 수: {len(all_surfaces)}")
				print(f"Input Surface 수: {len(input_surfaces)}")
				print(f"Output Surface 수: {len(output_surfaces)}")
				print(f"Input Surface ID: {sorted(input_surfaces)}")
				return

			# Option 2에서는 Input과 Output 사이에 들어가지 않은 나머지가
			# Insulated Surface입니다.
			insulated_surfaces = all_surfaces - input_surfaces - output_surfaces
			print("\n--- Option 2 확정 ---")
			print(f"전체 외부 Surface 수: {len(all_surfaces)}")
			print(f"Input Surface 수: {len(input_surfaces)}")
			print(f"Output Surface 수: {len(output_surfaces)}")
			print(f"Insulated Surface 수: {len(insulated_surfaces)}")
			print(f"Input Surface ID 목록: {sorted(input_surfaces)}")
			print(f"Output Surface ID 목록: {sorted(output_surfaces)}")

		def clear_input_selection():
			"""Input과 Output 선택을 모두 해제하고 actor 색상을 되돌립니다."""
			nonlocal output_surfaces
			input_surfaces.clear()
			output_surfaces.clear()
			for actor in actor_by_surface_id.values():
				actor.GetProperty().SetColor(0.83, 0.83, 0.83)
			plotter.render()
			print("\nInput Surface 선택을 모두 초기화했습니다.")
			print("현재 Input Surface 수: 0")
			print("현재 Input Surface ID 목록: []")

		# 왼쪽 마우스 클릭으로 actor를 선택합니다.
		# PyVista의 선택 강조용 임시 색상 대신 callback에서 빨간색을 유지합니다.
		plotter.enable_mesh_picking(
			callback=on_surface_picked,
			show=True,
			show_message=False,
			left_clicking=True,
			use_actor=True,
			picker="hardware",
		)
		# PyVista 창에서 Return 키는 Enter 키 역할을 합니다.
		# c 키를 누르면 선택한 Input Surface를 모두 초기화합니다.
		plotter.add_key_event("Return", confirm_option_1)
		plotter.add_key_event("c", clear_input_selection)
		plotter.add_key_event("1", set_option_1)
		plotter.add_key_event("2", set_option_2)
		plotter.add_key_event("i", select_input)
		plotter.add_key_event("o", select_output)

		show_boundary_mode()
		print("\nPyVista Viewer에서 외부 Surface를 클릭하세요.")
		print("이미 선택된 면을 다시 클릭하면 선택 해제됩니다.")
		print("1: Option 1, 2: Option 2")
		print("Option 1 - Enter: Input을 확정하고 나머지를 Output으로 지정합니다.")
		print("Option 2 - I: Input 선택, O: Output 선택, Enter: 확정")
		print("C: Input과 Output Surface 선택을 모두 초기화합니다.")
		print("Viewer 창을 닫으면 프로그램이 종료됩니다.")
		plotter.show()

	finally:
		# Viewer가 닫히거나 중간에 오류가 나도 Gmsh를 정리합니다.
		if gmsh.isInitialized():
			gmsh.finalize()


if __name__ == "__main__":
	main()
