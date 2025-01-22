import itertools
import json
import numpy as np
import pymesh
import pyvista as pv
import scipy
import trimesh
from collections import Counter

from point2cad.utils import suppress_output_fd


def save_unclipped_meshes(meshes, color_list, out_path):
    non_clipped_meshes = []
    pm_meshes = []
    for s in range(len(meshes)):
        tri_meshes_s = trimesh.Trimesh(
            vertices=np.array(meshes[s]["mesh"].points),
            faces=np.array(meshes[s]["mesh"].faces.reshape(-1, 4)[:, 1:]),
        )
        tri_meshes_s.visual.face_colors = color_list[s]
        non_clipped_meshes.append(tri_meshes_s)
        pm_meshes.append(
            pymesh.form_mesh(
                meshes[s]["mesh"].points,
                meshes[s]["mesh"].faces.reshape(-1, 4)[:, 1:],
            )
        )

    final_non_clipped = trimesh.util.concatenate(non_clipped_meshes)
    final_non_clipped.export(out_path)

    return pm_meshes


def save_clipped_meshes(pm_meshes, out_meshes, color_list, out_path):
    # pm_meshes是一个包含多个pymesh网格对象的列表
    # out_meshes是一个列表，其中每个元素是一个字典，用于切割操作
    # color_list是一个颜色列表，提供了每个输出网格的颜色
    # out_path是输出路径，指定最终导出的网格保存的位置

    # pymesh.merge_meshes()将所有的网格合并成一个网格，合并后的网格包含了所有输入网格的几何信息
    pm_merged = pymesh.merge_meshes(pm_meshes)
    # 获取合并网格中的“面来源”属性，并将其转换为整数类型
    face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)
    # 检测合并后的网格是否有自交，如果有，解决自交问题，得到一个没有自交的网格
    detect_pairs = pymesh.detect_self_intersection(pm_merged)
    pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)
    # 将解决了自交的网格分割成多个独立的子网格
    a = pymesh.separate_mesh(pm_resolved_ori)
    # 去除网格中误差在1e-6范围内的重复顶点
    pm_resolved, info_dict = pymesh.remove_duplicated_vertices(
        pm_resolved_ori, tol=1e-6, importance=None
    )
    # 获取去重复后网格的“面来源”属性
    face_sources_resolved_ori = pm_resolved_ori.get_attribute("face_sources").astype(
        np.int32
    )
    face_sources_from_fit = face_sources_merged[face_sources_resolved_ori]
    #使用trimesh库创建一个Trimesh对象，表示去重后的网格
    tri_resolved = trimesh.Trimesh(
        vertices=pm_resolved.vertices, faces=pm_resolved.faces
    )
    # 获取网格的面邻接关系
    face_adjacency = tri_resolved.face_adjacency
    # 获取网格的连通组件标签（每个标签表示一个连通的子网格）
    connected_node_labels = trimesh.graph.connected_component_labels(
        edges=face_adjacency, node_count=len(tri_resolved.faces)
    )
    # 使用Counter统计连通组件的频率，并返回最常见的连通组件ID
    most_common_groupids = [
        item[0] for item in Counter(connected_node_labels).most_common()
    ]
    # 基于连通组件ID，为每个连通的子网格创建一个独立的Trimesh对象
    submeshes = [
        trimesh.Trimesh(
            vertices=np.array(tri_resolved.vertices),
            faces=np.array(tri_resolved.faces)[np.where(connected_node_labels == item)],
        )
        for item in most_common_groupids
    ]
    # 提取每个子网格对应的面来源
    indices_sources = [
        face_sources_from_fit[connected_node_labels == item][0]
        for item in np.array(most_common_groupids)
    ]

    # 对每个输出网格，找出它的“inpoints”（网格内的点），并将这些点与子网格进行对比，找到与它们最接近的子网格
    # 计算每个子网格的面积，并根据面积选择与之最相关的子网格
    clipped_meshes = []
    further_clipped_meshes = []
    for p in range(len(out_meshes)):
        one_cluter_points = out_meshes[p]["inpoints"]
        submeshes_cur = [
            x
            for x, y in zip(submeshes, np.array(indices_sources) == p)
            if y and len(x.faces) > 2
        ]
        nearest_submesh = np.argmin(
            np.array(
                [
                    trimesh.proximity.closest_point(item, one_cluter_points)[1]
                    for item in submeshes_cur
                ]
            ).transpose(),
            -1,
        )
        counter_nearest = Counter(nearest_submesh).most_common()
        area_per_point = np.array(
            [submeshes_cur[item[0]].area / item[1] for item in counter_nearest]
        )

        multiplier_area = 2
        result_indices = np.array(counter_nearest)[:, 0][
            np.logical_and(
                area_per_point
                < area_per_point[np.nonzero(area_per_point)[0][0]] * multiplier_area,
                area_per_point != 0,
            )
        ]

        result_submesh_list = [submeshes_cur[item] for item in result_indices]
        # 将选定的子网格拼接成一个新的网格，并给该网格指定颜色
        clipped_mesh = trimesh.util.concatenate(result_submesh_list)
        clipped_mesh.visual.face_colors = color_list[p]
        clipped_meshes.append(clipped_mesh)
    # 将所有切割后的网格拼接在一起，导出到指定路径out_path
    clipped = trimesh.util.concatenate(clipped_meshes)
    clipped.export(out_path)
    # 返回所有切割后的子网格列表
    return clipped_meshes


def save_topology(clipped_meshes, out_path):
    filtered_submeshes_pv = [pv.wrap(item) for item in clipped_meshes]

    filtered_submeshes_pv_combinations = list(
        itertools.combinations(filtered_submeshes_pv, 2)
    )
    intersected_pair_indices = []
    intersection_curves = []
    intersections = {}

    for k, pv_pair in enumerate(filtered_submeshes_pv_combinations):
        with suppress_output_fd():
            intersection, _, _ = pv_pair[0].intersection(
                pv_pair[1], split_first=False, split_second=False, progress_bar=False
            )
        if intersection.n_points > 0:
            intersected_pair_indices.append(k)
            intersection_curve = {}
            intersection_curve["pv_points"] = intersection.points.tolist()
            intersection_curve["pv_lines"] = intersection.lines.reshape(-1, 3)[
                :, 1:
            ].tolist()
            intersection_curves.append(intersection_curve)

    intersections["curves"] = intersection_curves

    intersection_corners = []
    intersection_curves_combinations_indices = list(
        itertools.combinations(range(len(intersection_curves)), 2)
    )
    for combination_indices in intersection_curves_combinations_indices:
        sample0 = np.array(intersection_curves[combination_indices[0]]["pv_points"])
        sample1 = np.array(intersection_curves[combination_indices[1]]["pv_points"])

        dists = scipy.spatial.distance.cdist(sample0, sample1)
        row_indices, col_indices = np.where(dists == 0)

        if len(row_indices) > 0 and len(col_indices) > 0:
            corners = [
                (sample0[item[0]] + sample1[item[1]]) / 2
                for item in zip(row_indices, col_indices)
            ]
            intersection_corners.extend(corners)

    intersections["corners"] = [arr.tolist() for arr in intersection_corners]

    with open(out_path, "w") as cf:
        json.dump(intersections, cf)
