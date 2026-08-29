"""실제 STEP Grid Solid의 3D tetrahedral Volume Mesh 생성 테스트입니다."""

from pathlib import Path

import gmsh


# 두께 방향에 배치할 목표 element 개수입니다.
# 값을 키우면 두께 방향 mesh가 더 촘촘해집니다.
THICKNESS_ELEMENTS_TARGET = 2

# 전체 모델에 적용할 mesh 크기의 하한입니다.
# STEP 치수와 무관한 고정 Grid 크기가 아니라, 너무 작은 mesh_size로
# element 수가 과도하게 늘어나는 것을 막기 위한 안전 제한입니다.
MAX_GLOBAL_MESH_DENSITY = 100


def calculate_model_dimensions():
	"""동기화된 전체 모델의 Bounding Box와 방향별 역할을 계산합니다."""
	# -1, -1은 특정 Entity가 아니라 현재 모델 전체를 의미합니다.
	min_x, min_y, min_z, max_x, max_y, max_z = gmsh.model.getBoundingBox(-1, -1)
	dimensions_by_axis = {
		"x": max_x - min_x,
		"y": max_y - min_y,
		"z": max_z - min_z,
	}

	# 길이가 작은 축부터 찾으면 가장 작은 축이 두께 방향 후보입니다.
	sorted_axes = sorted(
		dimensions_by_axis,
		key=dimensions_by_axis.get,
	)
	return {
		**dimensions_by_axis,
		"thickness_axis": sorted_axes[0],
		"thickness": dimensions_by_axis[sorted_axes[0]],
	}


def print_geometry_information(volume_count, bounding_box, dimensions):
	"""STEP geometry 분석 결과를 Terminal에 출력합니다."""
	print("=== STEP Geometry Information ===")
	print(f"Volume count: {volume_count}")
	print(
		"Overall Bounding Box: "
		f"({bounding_box[0]:.3f}, {bounding_box[1]:.3f}, {bounding_box[2]:.3f}) - "
		f"({bounding_box[3]:.3f}, {bounding_box[4]:.3f}, {bounding_box[5]:.3f})"
	)
	print(f"X size: {dimensions['x']:.3f} mm")
	print(f"Y size: {dimensions['y']:.3f} mm")
	print(f"Z size: {dimensions['z']:.3f} mm")
	print(f"Detected thickness axis: {dimensions['thickness_axis'].upper()}")
	print(f"Detected thickness: {dimensions['thickness']:.3f} mm")


def print_mesh_element_information():
	"""생성된 3D element type과 개수를 확인해 출력합니다."""
	element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
	total_3d_elements = sum(len(tags) for tags in element_tags)

	print("\nElement types:")
	tetrahedral_element_count = 0
	for element_type, tags in zip(element_types, element_tags):
		element_name, _, _, number_of_nodes, _, _ = gmsh.model.mesh.getElementProperties(
			element_type
		)
		element_count = len(tags)
		print(
			f"Type {element_type}: {element_name}, "
			f"nodes/element: {number_of_nodes}, count: {element_count}"
		)
		if element_name.lower() == "tetrahedron" or number_of_nodes == 4:
			tetrahedral_element_count += element_count

	return total_3d_elements, tetrahedral_element_count


def main():
	"""STEP를 불러오고 3D Volume Mesh를 생성한 뒤 Gmsh GUI를 실행합니다."""
	step_path = Path(__file__).resolve().parent / "grid.step"
	if not step_path.exists():
		print(f"STEP load failed: file not found: {step_path}")
		return
	if THICKNESS_ELEMENTS_TARGET <= 0:
		print("Mesh setup failed: THICKNESS_ELEMENTS_TARGET must be greater than 0.")
		return

	mesh_created = False
	try:
		print(f"Loading STEP: {step_path}")
		gmsh.initialize()
		gmsh.option.setNumber("General.Terminal", 1)
		gmsh.model.add("grid_volume_mesh_test")

		print("STEP import: starting...")
		gmsh.model.occ.importShapes(str(step_path))
		gmsh.model.occ.synchronize()

		volume_entities = gmsh.model.getEntities(3)
		if not volume_entities:
			raise ValueError("No 3D volume was found after STEP import.")
		dimensions = calculate_model_dimensions()
		bounding_box = gmsh.model.getBoundingBox(-1, -1)
		print_geometry_information(len(volume_entities), bounding_box, dimensions)

		# 두께 / 목표 element 수를 기본 mesh 크기 후보로 사용합니다.
		mesh_size = dimensions["thickness"] / THICKNESS_ELEMENTS_TARGET
		# 너무 작은 mesh_size가 전체 모델로 퍼지지 않도록 전체 크기 기준의
		# 안전한 상한을 계산합니다. 실제 형상이나 치수를 수정하지 않습니다.
		largest_dimension = max(dimensions["x"], dimensions["y"], dimensions["z"])
		minimum_safe_global_size = largest_dimension / MAX_GLOBAL_MESH_DENSITY
		global_mesh_size = max(mesh_size, minimum_safe_global_size)
		gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
		gmsh.option.setNumber("Mesh.MeshSizeMax", global_mesh_size)
		gmsh.option.setNumber("Mesh.ElementOrder", 1)

		print("\n3D Volume Mesh: generating...")
		gmsh.model.mesh.generate(3)
		mesh_created = True

		node_tags, _, _ = gmsh.model.mesh.getNodes()
		total_3d_elements, tetrahedral_element_count = print_mesh_element_information()
		print("\n=== 3D Volume Mesh Result ===")
		print(f"STEP file: {step_path}")
		print(f"Volume count: {len(volume_entities)}")
		print("\nModel size:")
		print(f"X: {dimensions['x']:.3f} mm")
		print(f"Y: {dimensions['y']:.3f} mm")
		print(f"Z: {dimensions['z']:.3f} mm")
		print(f"Detected thickness axis: {dimensions['thickness_axis'].upper()}")
		print(f"Detected thickness: {dimensions['thickness']:.3f} mm")
		print(f"Target thickness elements: {THICKNESS_ELEMENTS_TARGET}")
		print(f"Target mesh size: {mesh_size:.3f} mm")
		print(f"Global mesh size limit: {global_mesh_size:.3f} mm")
		print(f"Total nodes: {len(node_tags)}")
		print(f"Total 3D elements: {total_3d_elements}")
		print(f"Tetrahedral elements: {tetrahedral_element_count}")

		if tetrahedral_element_count == 0:
			raise ValueError("3D mesh exists, but no tetrahedral elements were found.")

		print("\nOpening Gmsh GUI. Close the GUI window to finish.")
		gmsh.fltk.run()
	except Exception as error:
		if mesh_created:
			print(f"3D volume mesh verification failed: {error}")
		else:
			print(f"3D volume mesh generation failed: {error}")
	finally:
		if gmsh.isInitialized():
			gmsh.finalize()


if __name__ == "__main__":
	main()
