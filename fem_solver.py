"""Gmsh 3D tetrahedral mesh를 이용한 DC 전기전도 FEM solver입니다."""

import math
import gmsh
import numpy as np
from skfem import Basis, BilinearForm, ElementTetP1, FacetBasis, Functional, LinearForm, MeshTet, asm, condense, solve
from skfem.helpers import dot, grad


def _extract_nodes():
    """Gmsh node를 scikit-fem용 좌표와 tag-index mapping으로 읽습니다."""
    node_tags, coordinates_mm, _ = gmsh.model.mesh.getNodes()
    if not len(node_tags):
        raise ValueError("Gmsh mesh에 node가 없습니다.")

    # Gmsh와 STEP의 좌표는 mm이지만 conductivity는 S/m입니다.
    # 따라서 FEM 좌표는 반드시 mm에서 m로 변환합니다.
    points_m = np.asarray(coordinates_mm, dtype=float).reshape((-1, 3)) * 1.0e-3
    node_index_by_tag = {
        int(node_tag): index for index, node_tag in enumerate(node_tags)
    }
    return points_m, node_index_by_tag


def _extract_tetrahedra(node_index_by_tag):
    """3D element 중 4-node first-order tetrahedron만 추출합니다."""
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(3)
    tetrahedra = []
    for element_type, tags, node_tags in zip(
        element_types, element_tags, element_node_tags
    ):
        _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(
            element_type
        )
        if nodes_per_element != 4:
            continue
        elements = np.asarray(node_tags, dtype=np.int64).reshape((-1, 4))
        for element in elements:
            try:
                tetrahedra.append(
                    [node_index_by_tag[int(node_tag)] for node_tag in element]
                )
            except KeyError as error:
                raise ValueError(
                    f"Tetrahedron node tag {error.args[0]}를 찾을 수 없습니다."
                ) from error

    if not tetrahedra:
        raise ValueError("4-node first-order tetrahedral element를 찾지 못했습니다.")
    return np.asarray(tetrahedra, dtype=np.int64)


def _extract_surface_triangles(surface_ids, node_index_by_tag):
    """Gmsh Surface의 3-node triangle을 정렬된 node triplet으로 반환합니다."""
    triangles = []
    for surface_id in sorted(surface_ids):
        element_types, _, element_node_tags = gmsh.model.mesh.getElements(2, surface_id)
        for element_type, node_tags in zip(element_types, element_node_tags):
            _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(
                element_type
            )
            if nodes_per_element != 3:
                continue
            elements = np.asarray(node_tags, dtype=np.int64).reshape((-1, 3))
            for element in elements:
                try:
                    triangles.append(
                        tuple(sorted(node_index_by_tag[int(node_tag)] for node_tag in element))
                    )
                except KeyError as error:
                    raise ValueError(
                        f"Surface {surface_id}의 node tag {error.args[0]}를 찾을 수 없습니다."
                    ) from error
    return triangles


def _map_surface_facets(mesh, surface_ids, node_index_by_tag):
    """Gmsh triangle node triplet을 scikit-fem facet index로 변환합니다."""
    facet_index_by_nodes = {
        tuple(sorted(int(node) for node in facet)): index
        for index, facet in enumerate(mesh.facets.T)
    }
    mapped_facets = {
        facet_index_by_nodes[triangle]
        for triangle in _extract_surface_triangles(surface_ids, node_index_by_tag)
        if triangle in facet_index_by_nodes
    }
    return np.asarray(sorted(mapped_facets), dtype=np.int64)


def _build_mesh_and_facet_data(input_surfaces, output_surfaces):
    """공통 Gmsh mesh와 Input/Output facet mapping을 준비합니다."""
    points_m, node_index_by_tag = _extract_nodes()
    tetrahedra = _extract_tetrahedra(node_index_by_tag)
    mesh = MeshTet(points_m.T, tetrahedra.T)
    input_facets = _map_surface_facets(mesh, input_surfaces, node_index_by_tag)
    output_facets = _map_surface_facets(mesh, output_surfaces, node_index_by_tag)
    if not len(input_facets) or not len(output_facets):
        raise ValueError("Input/Output Surface를 FEM facet으로 매핑하지 못했습니다.")
    return points_m, tetrahedra, mesh, node_index_by_tag, input_facets, output_facets


def _calculate_flux_current(mesh, element, solution, conductivity, input_facets):
    """Input FacetBasis에서 normal flux를 적분하는 보조 검증입니다."""
    if not len(input_facets):
        return None
    try:
        facet_basis = FacetBasis(mesh, element, facets=input_facets)

        @Functional
        def input_flux(w):
            # J = -sigma * grad(V), 그리고 n 방향 성분을 적분합니다.
            return -conductivity * dot(w.n, grad(w.u))

        flux = float(input_flux.assemble(facet_basis.interpolate(solution)))
        return abs(flux)
    except Exception as error:
        print(f"Flux verification error: {error}")
        return None


def _calculate_post_processing(points_m, tetrahedra, solution, conductivity, energy_power):
    """P1 tetrahedron별 E, J, Joule loss와 hotspot을 계산합니다."""
    electric_field = np.empty((len(tetrahedra), 3), dtype=float)
    current_density = np.empty((len(tetrahedra), 3), dtype=float)
    tetrahedron_volumes = np.empty(len(tetrahedra), dtype=float)

    for element_index, element_nodes in enumerate(tetrahedra):
        coordinates = points_m[element_nodes]
        potentials = solution[element_nodes]
        # V = a + b*x + c*y + d*z를 만족하는 계수를 구합니다.
        coefficient_matrix = np.column_stack(
            [np.ones(4), coordinates]
        )
        try:
            coefficients = np.linalg.solve(coefficient_matrix, potentials)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"Tetrahedron {element_index}의 potential gradient를 계산할 수 없습니다."
            ) from error

        determinant = np.linalg.det(
            np.column_stack(
                [coordinates[1] - coordinates[0], coordinates[2] - coordinates[0], coordinates[3] - coordinates[0]]
            )
        )
        volume = abs(float(determinant)) / 6.0
        if not math.isfinite(volume) or volume <= np.finfo(float).eps:
            raise ValueError(f"Tetrahedron {element_index}의 volume이 유효하지 않습니다.")
        tetrahedron_volumes[element_index] = volume
        electric_field[element_index] = -coefficients[1:]
        current_density[element_index] = conductivity * electric_field[element_index]

    current_density_magnitude = np.linalg.norm(current_density, axis=1)
    electric_field_magnitude = np.linalg.norm(electric_field, axis=1)
    joule_loss_density = np.einsum("ij,ij->i", current_density, electric_field)
    if not np.all(np.isfinite(joule_loss_density)):
        raise ValueError("Joule loss density에 NaN 또는 infinite 값이 포함되어 있습니다.")
    if np.any(joule_loss_density < -1.0e-10 * max(1.0, float(np.max(np.abs(joule_loss_density))))):
        raise ValueError("Joule loss density에 명백한 음수 값이 있습니다.")
    joule_loss_density = np.maximum(joule_loss_density, 0.0)
    integrated_joule_power = float(np.sum(joule_loss_density * tetrahedron_volumes))
    power_difference_percent = (
        abs(integrated_joule_power - energy_power) / abs(energy_power) * 100.0
        if energy_power != 0
        else None
    )
    if power_difference_percent is None or power_difference_percent > 1.0e-5:
        raise ValueError(
            "Integrated Joule Power가 FEM energy power와 일치하지 않습니다. "
            f"차이: {power_difference_percent:.6g} %"
        )

    hotspot_element_index = int(np.argmax(joule_loss_density))
    hotspot_centroid_mm = (
        np.mean(points_m[tetrahedra[hotspot_element_index]], axis=0) * 1000.0
    )
    return {
        "electric_field_vector": electric_field,
        "electric_field_magnitude": electric_field_magnitude,
        "current_density_vector": current_density,
        "current_density_magnitude": current_density_magnitude,
        "joule_loss_density": joule_loss_density,
        "tetrahedron_volumes_m3": tetrahedron_volumes,
        "integrated_joule_power_w": integrated_joule_power,
        "joule_power_difference_percent": power_difference_percent,
        "hotspot": {
            "element_index": hotspot_element_index,
            "joule_loss_density_w_m3": float(joule_loss_density[hotspot_element_index]),
            "current_density_a_m2": float(current_density_magnitude[hotspot_element_index]),
            "electric_field_v_m": float(electric_field_magnitude[hotspot_element_index]),
            "centroid_mm": hotspot_centroid_mm.tolist(),
        },
    }
def run_electrical_fem(input_surfaces, output_surfaces, conductivity):
    """현재 Gmsh mesh와 Surface ID로 DC conduction FEM을 실행합니다."""
    if conductivity is None or not math.isfinite(float(conductivity)) or conductivity <= 0:
        raise ValueError("Electrical conductivity는 0보다 큰 유한한 값이어야 합니다.")
    if not input_surfaces or not output_surfaces:
        raise ValueError("Input과 Output Surface가 각각 하나 이상 필요합니다.")

    points_m, node_index_by_tag = _extract_nodes()
    tetrahedra = _extract_tetrahedra(node_index_by_tag)
    mesh = MeshTet(points_m.T, tetrahedra.T)
    input_facets = _map_surface_facets(mesh, input_surfaces, node_index_by_tag)
    output_facets = _map_surface_facets(mesh, output_surfaces, node_index_by_tag)
    if not len(input_facets):
        raise ValueError("Gmsh Input Surface를 FEM facet으로 매핑하지 못했습니다.")
    if not len(output_facets):
        raise ValueError("Gmsh Output Surface를 FEM facet으로 매핑하지 못했습니다.")
    if np.intersect1d(input_facets, output_facets).size:
        raise ValueError("Input과 Output FEM facet이 겹칩니다.")

    print(f"Gmsh Input Surfaces: {sorted(input_surfaces)}")
    print(f"Mapped Input FEM Facets: {input_facets.tolist()}")
    print(f"Gmsh Output Surfaces: {sorted(output_surfaces)}")
    print(f"Mapped Output FEM Facets: {output_facets.tolist()}")

    mesh = mesh.with_boundaries({"Input": input_facets, "Output": output_facets})
    element = ElementTetP1()
    basis = Basis(mesh, element)

    @BilinearForm
    def conduction(u, v, w):
        return conductivity * dot(grad(u), grad(v))

    matrix = asm(conduction, basis)
    dirichlet_dofs = np.unique(
        np.concatenate([basis.get_dofs("Input").all(), basis.get_dofs("Output").all()])
    )
    prescribed = np.zeros(basis.N)
    prescribed[basis.get_dofs("Input").all()] = 1.0
    solution = solve(*condense(matrix, np.zeros(basis.N), x=prescribed, D=dirichlet_dofs))
    if not np.all(np.isfinite(solution)):
        raise ValueError("FEM potential에 NaN 또는 infinite 값이 포함되어 있습니다.")

    delta_v = 1.0
    power = float(solution @ (matrix @ solution))
    conductance = power / (delta_v**2)
    current = conductance * delta_v
    if not math.isfinite(conductance) or conductance <= 0 or current == 0:
        raise ValueError("Conductance 또는 current가 유효하지 않습니다.")
    resistance_ohm = delta_v / current
    flux_current = _calculate_flux_current(
        mesh, element, solution, conductivity, input_facets
    )
    flux_difference_percent = (
        abs(flux_current - current) / abs(current) * 100.0
        if flux_current is not None
        else None
    )
    post_processing = _calculate_post_processing(
        points_m, tetrahedra, solution, conductivity, power
    )

    return {
        "success": True,
        "potential": solution,
        "dofs": basis.N,
        "min_potential_v": float(np.min(solution)),
        "max_potential_v": float(np.max(solution)),
        "delta_v": delta_v,
        "conductance_s": conductance,
        "current_a": current,
        "resistance_ohm": resistance_ohm,
        "resistance_mohm": resistance_ohm * 1.0e3,
        "resistance_uohm": resistance_ohm * 1.0e6,
        "flux_current_a": flux_current,
        "flux_difference_percent": flux_difference_percent,
        "mesh_nodes": len(points_m),
        "tetrahedra": len(tetrahedra),
        "input_facets": len(input_facets),
        "output_facets": len(output_facets),
        "mesh_points_m": points_m,
        "tetrahedra_connectivity": tetrahedra,
        **post_processing,
    }


def scale_fem_result_to_current(fem_result, analysis_current_a, reference_potential_v=2.14):
    """1 V 결과를 실제 전류와 절대전위 기준으로 표시용 scaling합니다."""
    analysis_current_a = float(analysis_current_a)
    normalized_current = float(fem_result["current_a"])
    resistance_ohm = float(fem_result["resistance_ohm"])
    reference_potential_v = float(reference_potential_v)
    if not math.isfinite(analysis_current_a) or analysis_current_a <= 0:
        raise ValueError("Analysis Current는 0보다 큰 유한한 값이어야 합니다.")
    if not math.isfinite(normalized_current) or normalized_current <= 0:
        raise ValueError("1 V normalized FEM current가 유효하지 않습니다.")
    if not math.isfinite(reference_potential_v):
        raise ValueError("Reference Potential은 유한한 숫자여야 합니다.")

    scale_factor = analysis_current_a / normalized_current
    voltage_drop = analysis_current_a * resistance_ohm
    power_actual = analysis_current_a**2 * resistance_ohm
    power_from_voltage = voltage_drop * analysis_current_a
    if not math.isclose(power_actual, power_from_voltage, rel_tol=1.0e-10):
        raise ValueError("실제 전류 power 계산이 일치하지 않습니다.")

    # 정상상태 선형 재료 문제에서는 V, E, J가 전류에 선형 비례하고,
    # Joule loss는 J dot E이므로 전류의 제곱에 비례합니다.
    actual_potential = np.asarray(fem_result["potential"]) * voltage_drop
    # Reference Potential은 constant offset이므로 grad(V)에 영향을 주지 않습니다.
    # 따라서 저항, 전류밀도, Joule loss는 변하지 않고 절대전위 표시만 이동합니다.
    voltage_drop_field = actual_potential
    absolute_potential = reference_potential_v - voltage_drop + voltage_drop_field
    actual_electric_field = np.asarray(fem_result["electric_field_vector"]) * scale_factor
    actual_current_density = np.asarray(fem_result["current_density_vector"]) * scale_factor
    actual_current_density_magnitude = (
        np.asarray(fem_result["current_density_magnitude"]) * abs(scale_factor)
    )
    actual_joule_loss = np.asarray(fem_result["joule_loss_density"]) * scale_factor**2
    actual_integrated_power = float(
        np.sum(actual_joule_loss * np.asarray(fem_result["tetrahedron_volumes_m3"]))
    )
    actual_power_difference = abs(actual_integrated_power - power_actual) / abs(power_actual) * 100.0
    hotspot_index = fem_result["hotspot"]["element_index"]
    actual_hotspot = dict(fem_result["hotspot"])
    actual_hotspot["joule_loss_density_w_m3"] = float(actual_joule_loss[hotspot_index])
    actual_hotspot["current_density_a_m2"] = float(actual_current_density_magnitude[hotspot_index])
    actual_hotspot["current_density_a_mm2"] = actual_hotspot["current_density_a_m2"] / 1.0e6
    return {
        "analysis_current_a": analysis_current_a,
        "reference_potential_v": reference_potential_v,
        "scale_factor": scale_factor,
        "voltage_drop_v": voltage_drop,
        "voltage_drop_field_v": voltage_drop_field,
        "absolute_potential_v": absolute_potential,
        "absolute_potential_min_v": float(np.min(absolute_potential)),
        "absolute_potential_max_v": float(np.max(absolute_potential)),
        "power_loss_w": power_actual,
        "actual_potential_v": actual_potential,
        "actual_electric_field_vector": actual_electric_field,
        "actual_current_density_vector": actual_current_density,
        "actual_current_density_magnitude": actual_current_density_magnitude,
        "actual_current_density_magnitude_a_mm2": actual_current_density_magnitude / 1.0e6,
        "actual_joule_loss_density": actual_joule_loss,
        "actual_integrated_joule_power_w": actual_integrated_power,
        "actual_power_difference_percent": actual_power_difference,
        "actual_hotspot": actual_hotspot,
    }


def run_real_battery_gaussian_fem(
    input_surfaces,
    sink_surfaces,
    conductivity,
    analysis_current_a,
    center_xy=(0.0, 0.0),
    spread_xy=(1.0, 1.0),
):
    """Lug와 Gaussian distributed active-area sink을 연결하는 별도 Grid 모델입니다."""
    if conductivity <= 0 or analysis_current_a <= 0:
        raise ValueError("conductivity와 Analysis Current는 0보다 커야 합니다.")
    points_m, tetrahedra, mesh, node_index_by_tag, input_facets, _ = (
        _build_mesh_and_facet_data(input_surfaces, sink_surfaces)
    )
    sink_facets = _map_surface_facets(mesh, sink_surfaces, node_index_by_tag)
    if not len(sink_facets):
        raise ValueError("Gaussian sink Surface를 FEM facet으로 매핑하지 못했습니다.")
    mesh = mesh.with_boundaries({"Lug": input_facets, "Sink": sink_facets})
    element = ElementTetP1()
    basis = Basis(mesh, element)

    center_m = np.asarray(center_xy, dtype=float) * 1.0e-3
    spread_m = np.asarray(spread_xy, dtype=float) * 1.0e-3
    if np.any(spread_m <= 0):
        raise ValueError("Gaussian Spread X/Y는 0보다 커야 합니다.")
    facet_basis = FacetBasis(mesh, element, facets=sink_facets)

    def gaussian_weight(x):
        return np.exp(-0.5 * (((x[0] - center_m[0]) / spread_m[0]) ** 2 + ((x[1] - center_m[1]) / spread_m[1]) ** 2))

    @Functional
    def gaussian_area(w):
        return gaussian_weight(w.x)

    normalization = float(gaussian_area.assemble(facet_basis))
    if normalization <= 0 or not math.isfinite(normalization):
        raise ValueError("Gaussian sink normalization을 계산하지 못했습니다.")

    @BilinearForm
    def conduction(u, v, w):
        return conductivity * dot(grad(u), grad(v))

    @LinearForm
    def sink_load(v, w):
        # Gaussian 분포를 facet 면적에 대해 정규화하여 전체 sink current가
        # Analysis Current와 같아지도록 합니다. 이는 electrochemical solver가
        # 아니라 active-material 전류 기여를 근사하는 engineering model입니다.
        return analysis_current_a * gaussian_weight(w.x) / normalization * v

    matrix = asm(conduction, basis)
    rhs = asm(sink_load, facet_basis)
    lug_dofs = basis.get_dofs("Lug").all()
    solution = solve(*condense(matrix, rhs, D=lug_dofs))
    if not np.all(np.isfinite(solution)):
        raise ValueError("Gaussian FEM potential에 NaN 또는 infinite 값이 있습니다.")
    energy_power = float(solution @ (matrix @ solution))
    post_processing = _calculate_post_processing(
        points_m, tetrahedra, solution, conductivity, energy_power
    )
    voltage_drop = float(np.max(solution) - np.min(solution))
    resistance = voltage_drop / analysis_current_a
    actual_hotspot = dict(post_processing["hotspot"])
    actual_hotspot["current_density_a_mm2"] = (
        actual_hotspot["current_density_a_m2"] / 1.0e6
    )
    return {
        "success": True,
        "potential": solution,
        "dofs": basis.N,
        "min_potential_v": float(np.min(solution)),
        "max_potential_v": float(np.max(solution)),
        "delta_v": voltage_drop,
        "conductance_s": 1.0 / resistance if resistance > 0 else None,
        "current_a": analysis_current_a,
        "resistance_ohm": resistance,
        "resistance_mohm": resistance * 1.0e3,
        "resistance_uohm": resistance * 1.0e6,
        "flux_current_a": None,
        "flux_difference_percent": None,
        "mesh_nodes": len(points_m),
        "tetrahedra": len(tetrahedra),
        "input_facets": len(input_facets),
        "output_facets": len(sink_facets),
        "mesh_points_m": points_m,
        "tetrahedra_connectivity": tetrahedra,
        "analysis_current_a": analysis_current_a,
        "scale_factor": 1.0,
        "voltage_drop_v": voltage_drop,
        "voltage_drop_field_v": solution,
        "actual_potential_v": solution,
        "actual_electric_field_vector": post_processing["electric_field_vector"],
        "actual_current_density_vector": post_processing["current_density_vector"],
        "actual_current_density_magnitude": post_processing["current_density_magnitude"],
        "actual_current_density_magnitude_a_mm2": (
            post_processing["current_density_magnitude"] / 1.0e6
        ),
        "actual_joule_loss_density": post_processing["joule_loss_density"],
        "actual_integrated_joule_power_w": post_processing["integrated_joule_power_w"],
        "power_loss_w": energy_power,
        "actual_hotspot": actual_hotspot,
        "gaussian_center_xy_mm": np.asarray(center_xy, dtype=float).tolist(),
        "gaussian_spread_xy_mm": np.asarray(spread_xy, dtype=float).tolist(),
        "integrated_sink_current_a": float(analysis_current_a),
        "sink_current_balance_error_percent": 0.0,
        **post_processing,
    }


def scale_real_battery_result_to_current(result, analysis_current_a):
    """Gaussian FEM 결과를 새 전류로 scaling하며 FEM solve는 다시 하지 않습니다."""
    analysis_current_a = float(analysis_current_a)
    old_current = float(result["current_a"])
    if analysis_current_a <= 0 or old_current <= 0:
        raise ValueError("Analysis Current는 0보다 커야 합니다.")
    scale_factor = analysis_current_a / old_current
    updated = dict(result)
    updated["analysis_current_a"] = analysis_current_a
    updated["current_a"] = analysis_current_a
    updated["scale_factor"] = scale_factor
    updated["potential"] = np.asarray(result["potential"]) * scale_factor
    updated["voltage_drop_field_v"] = np.asarray(result["voltage_drop_field_v"]) * scale_factor
    updated["actual_potential_v"] = updated["potential"]
    updated["actual_electric_field_vector"] = np.asarray(result["actual_electric_field_vector"]) * scale_factor
    updated["actual_current_density_vector"] = np.asarray(result["actual_current_density_vector"]) * scale_factor
    updated["actual_current_density_magnitude"] = np.asarray(result["actual_current_density_magnitude"]) * abs(scale_factor)
    updated["actual_current_density_magnitude_a_mm2"] = updated["actual_current_density_magnitude"] / 1.0e6
    updated["actual_joule_loss_density"] = np.asarray(result["actual_joule_loss_density"]) * scale_factor**2
    updated["delta_v"] = float(result["delta_v"]) * abs(scale_factor)
    updated["voltage_drop_v"] = float(result["voltage_drop_v"]) * abs(scale_factor)
    updated["power_loss_w"] = float(result["power_loss_w"]) * scale_factor**2
    updated["actual_integrated_joule_power_w"] = float(result["actual_integrated_joule_power_w"]) * scale_factor**2
    updated["actual_hotspot"] = dict(result["actual_hotspot"])
    updated["actual_hotspot"]["current_density_a_m2"] *= abs(scale_factor)
    updated["actual_hotspot"]["current_density_a_mm2"] *= abs(scale_factor)
    updated["actual_hotspot"]["joule_loss_density_w_m3"] *= scale_factor**2
    return updated
