from __future__ import annotations

import numpy as np
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def terrain_path_from_motion(motion, override: str | None = None) -> Path | None:
    if override:
        return Path(override).expanduser()
    if "terrain_stl" not in motion:
        return None
    raw = motion["terrain_stl"]
    value = raw.item() if isinstance(raw, np.ndarray) and raw.shape == () else raw
    value = str(value).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(motion.path).expanduser().parent / path
    return path


def mujoco_xml_with_terrain_mesh(
    xml_path: str | Path,
    terrain_path: str | Path | None,
    *,
    mode: str = "auto",
    collision_backend: str = "boxes",
    mesh_name: str = "wbc_terrain_mesh",
    geom_name: str = "wbc_terrain",
    geom_group: int = 0,
    visual_group: int = 2,
    rgba: tuple[float, float, float, float] = (0.45, 0.38, 0.28, 1.0),
    friction: tuple[float, float, float] = (0.8, 0.8, 0.005),
) -> tuple[str, tempfile.TemporaryDirectory[str] | None, Path | None]:
    """Return an XML path that optionally contains a MuJoCo STL terrain mesh.

    The original robot XML is left untouched. When mesh terrain is enabled, a temporary XML is
    generated with one extra mesh asset and one world geom.
    """

    mode = str(mode)
    if mode not in ("auto", "flat", "mesh"):
        raise ValueError(f"Unsupported terrain mode {mode!r}; expected 'auto', 'flat', or 'mesh'.")
    collision_backend = str(collision_backend)
    if collision_backend not in ("mesh", "boxes"):
        raise ValueError(f"Unsupported terrain collision backend {collision_backend!r}; expected 'mesh' or 'boxes'.")
    if mode == "flat":
        return str(xml_path), None, None
    if terrain_path is None:
        if mode == "mesh":
            raise ValueError("--terrain_mode mesh requires a terrain STL from --terrain_stl, manifest, or npz.")
        return str(xml_path), None, None

    terrain_path = Path(terrain_path).expanduser().resolve()
    if not terrain_path.is_file():
        raise FileNotFoundError(f"Terrain STL not found: {terrain_path}")

    xml_path = Path(xml_path).expanduser().resolve()
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(str(compiler.get("meshdir"))).expanduser()
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((xml_path.parent / meshdir).resolve()))
    assets = root.findall("asset")
    if assets:
        asset = assets[-1]
    else:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(terrain_path)})

    worldbodies = root.findall("worldbody")
    if worldbodies:
        worldbody = worldbodies[-1]
    else:
        worldbody = ET.SubElement(root, "worldbody")
    if collision_backend == "mesh":
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": geom_name,
                "type": "mesh",
                "mesh": mesh_name,
                "group": str(int(geom_group)),
                "contype": "1",
                "conaffinity": "1",
                "rgba": _float_tuple(rgba),
                "friction": _float_tuple(friction),
            },
        )
    else:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"{geom_name}_visual",
                "type": "mesh",
                "mesh": mesh_name,
                "group": str(int(visual_group)),
                "contype": "0",
                "conaffinity": "0",
                "rgba": _float_tuple(rgba),
            },
        )
        boxes = terrain_boxes_from_stl(terrain_path)
        if not boxes:
            raise ValueError(f"Could not extract any collision boxes from terrain STL: {terrain_path}")
        for box_id, box in enumerate(boxes):
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": f"{geom_name}_box_{box_id:03d}",
                    "type": "box",
                    "pos": _float_tuple(box["pos"]),
                    "size": _float_tuple(box["size"]),
                    "euler": f"0 0 {float(box['yaw']):.9g}",
                    "group": str(int(geom_group)),
                    "contype": "1",
                    "conaffinity": "1",
                    "rgba": "0.25 0.8 0.25 0.25",
                    "friction": _float_tuple(friction),
                },
            )
        print(f"[INFO] Extracted {len(boxes)} MuJoCo terrain collision boxes from {terrain_path}")

    tmpdir = tempfile.TemporaryDirectory(prefix="wbc_mujoco_terrain_")
    out_path = Path(tmpdir.name) / Path(xml_path).name
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return str(out_path), tmpdir, terrain_path


def _float_tuple(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def terrain_boxes_from_stl(
    terrain_path: str | Path,
    *,
    normal_z_threshold: float = 0.5,
    min_height: float = 1e-4,
    min_area: float = 1e-5,
) -> list[dict[str, object]]:
    """Extract oriented box collision geoms from horizontal top faces in an STL terrain."""

    import trimesh

    import shapely.geometry as geom
    import shapely.ops as ops

    mesh = trimesh.load(str(terrain_path), force="mesh", process=False)
    if mesh.is_empty:
        return []
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)

    top_polys_by_z: dict[float, list] = {}
    for face, normal in zip(faces, normals):
        tri = vertices[face]
        if normal[2] < normal_z_threshold:
            continue
        if float(np.ptp(tri[:, 2])) > 1e-5:
            continue
        poly = geom.Polygon(tri[:, :2])
        if not poly.is_valid or poly.area <= min_area:
            continue
        z = round(float(np.mean(tri[:, 2])), 6)
        top_polys_by_z.setdefault(z, []).append(poly)

    boxes: list[dict[str, object]] = []
    for top_z, polys in sorted(top_polys_by_z.items()):
        merged = ops.unary_union(polys)
        if merged.geom_type == "Polygon":
            top_regions = [merged]
        else:
            top_regions = list(getattr(merged, "geoms", []))
        for region in top_regions:
            if region.area <= min_area:
                continue
            bottom_z = _infer_box_bottom_z(vertices, faces, region, top_z)
            if top_z - bottom_z < min_height:
                continue
            box = _oriented_box_from_top_region(region, bottom_z, top_z)
            if box is not None:
                boxes.append(box)
    return boxes


def _infer_box_bottom_z(vertices: np.ndarray, faces: np.ndarray, region, top_z: float) -> float:
    import shapely.geometry as geom

    min_z = float(np.min(vertices[:, 2]))
    region_search = region.buffer(1e-5)
    candidates = []
    for face in faces:
        tri = vertices[face]
        if float(np.max(tri[:, 2])) < top_z - 1e-5:
            continue
        if float(np.min(np.abs(tri[:, 2] - top_z))) > 1e-5:
            continue
        xy = tri[:, :2]
        primitive = geom.Polygon(xy)
        if not primitive.is_valid or primitive.area <= 1e-12:
            primitive = geom.LineString(xy)
        if region_search.intersects(primitive):
            candidates.append(float(np.min(tri[:, 2])))
    if candidates:
        return min(candidates)
    return min_z


def _oriented_box_from_top_region(region, bottom_z: float, top_z: float) -> dict[str, object] | None:
    rect = region.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords[:-1], dtype=np.float64)
    if coords.shape != (4, 2):
        return None
    edges = np.roll(coords, -1, axis=0) - coords
    lengths = np.linalg.norm(edges, axis=1)
    first = int(np.argmax(lengths))
    second = (first + 1) % 4
    length_x = float(lengths[first])
    length_y = float(lengths[second])
    if length_x <= 1e-6 or length_y <= 1e-6:
        return None
    yaw = float(np.arctan2(edges[first, 1], edges[first, 0]))
    center_xy = np.mean(coords, axis=0)
    height = float(top_z - bottom_z)
    return {
        "pos": (float(center_xy[0]), float(center_xy[1]), float(bottom_z + 0.5 * height)),
        "size": (0.5 * length_x, 0.5 * length_y, 0.5 * height),
        "yaw": yaw,
    }


def _yaw_rotation_from_body_xmat(body_xmat: np.ndarray) -> np.ndarray:
    body_xmat = np.asarray(body_xmat, dtype=np.float64).reshape(3, 3)
    yaw = float(np.arctan2(body_xmat[1, 0], body_xmat[0, 0]))
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.asarray([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def mj_ray_height_scan(
    model,
    data,
    *,
    body_id: int,
    local_points: np.ndarray,
    offset: np.ndarray,
    ground_z: float = 0.0,
    height_offset: float = 0.5,
    clip: tuple[float, float] = (-2.0, 2.0),
    geom_groups: tuple[int, ...] = (0,),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return IsaacLab-style height_scan values using MuJoCo ray casting."""

    import mujoco

    body_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
    rot = _yaw_rotation_from_body_xmat(data.xmat[body_id])
    offset = np.asarray(offset, dtype=np.float64)
    local_points = np.asarray(local_points, dtype=np.float64)

    scan_origin = body_pos + rot @ offset
    ray_starts = scan_origin[None, :] + local_points @ rot.T
    ray_dir = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    geomgroup = np.zeros(6, dtype=np.uint8)
    for group in geom_groups:
        group = int(group)
        if group < 0 or group >= geomgroup.shape[0]:
            raise ValueError(f"MuJoCo geom group must be in [0, 5], got {group}.")
        geomgroup[group] = 1
    hit_points = np.empty_like(ray_starts)
    heights = np.empty(ray_starts.shape[0], dtype=np.float64)
    geom_ids = np.full(ray_starts.shape[0], -1, dtype=np.int32)

    for i, ray_start in enumerate(ray_starts):
        geom_id = np.asarray([-1], dtype=np.int32)
        dist = float(mujoco.mj_ray(model, data, ray_start, ray_dir, geomgroup, 1, int(body_id), geom_id))
        if dist >= 0.0:
            hit = ray_start + dist * ray_dir
            geom_ids[i] = int(geom_id[0])
        else:
            hit = np.asarray([ray_start[0], ray_start[1], ground_z], dtype=np.float64)
        hit_points[i] = hit
        heights[i] = float(hit[2])

    values = scan_origin[2] - offset[2] - heights - float(height_offset)
    values = np.clip(values, clip[0], clip[1]).reshape(1, -1).astype(np.float32)
    cache = {
        "scan_origin": scan_origin.astype(np.float64),
        "ray_starts": ray_starts.astype(np.float64),
        "hit_points": hit_points.astype(np.float64),
        "values": values.copy(),
        "geom_ids": geom_ids,
    }
    return values, cache


class MujocoHeightScanner:
    """IsaacLab-style yaw-aligned terrain height scan for MuJoCo deploy."""

    def __init__(
        self,
        model,
        *,
        body_name: str = "torso_link",
        terrain_path: Path | None = None,
        offset: tuple[float, float, float] = (0.80, 0.0, 20.0),
        size: tuple[float, float] = (1.6, 1.0),
        resolution: float = 0.1,
        ground_z: float = 0.0,
        height_offset: float = 0.5,
        clip: tuple[float, float] = (-2.0, 2.0),
        backend: str = "mj_ray",
        geom_groups: tuple[int, ...] = (0,),
    ) -> None:
        self.body_name = body_name
        self.model = model
        self.body_id = int(model.body(body_name).id)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.ground_z = float(ground_z)
        self.height_offset = float(height_offset)
        self.clip = clip
        self.backend = str(backend)
        self.geom_groups = tuple(int(group) for group in geom_groups)
        self.mesh = None
        self.terrain_path = Path(terrain_path).expanduser() if terrain_path is not None else None
        self.local_points = self._grid_points(size=size, resolution=resolution)
        self.last_scan: dict[str, np.ndarray] | None = None
        if self.backend not in ("mj_ray", "trimesh"):
            raise ValueError(f"Unsupported height scan backend: {self.backend!r}")
        if self.terrain_path is not None and self.backend == "trimesh":
            if not self.terrain_path.is_file():
                raise FileNotFoundError(self.terrain_path)
            import trimesh

            mesh = trimesh.load(self.terrain_path, force="mesh", process=False)
            if mesh.is_empty:
                raise ValueError(f"{self.terrain_path}: terrain mesh is empty.")
            self.mesh = mesh

    @staticmethod
    def _grid_points(*, size: tuple[float, float], resolution: float) -> np.ndarray:
        xs = np.arange(-size[0] * 0.5, size[0] * 0.5 + resolution * 0.5, resolution, dtype=np.float64)
        ys = np.arange(-size[1] * 0.5, size[1] * 0.5 + resolution * 0.5, resolution, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        return np.stack((xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size, dtype=np.float64)), axis=-1)

    @property
    def dim(self) -> int:
        return int(self.local_points.shape[0])

    def print_config(self) -> None:
        if self.backend == "mj_ray":
            source = f"MuJoCo mj_ray with fallback flat ground z={self.ground_z:.3f}"
        else:
            source = str(self.terrain_path) if self.terrain_path is not None else f"flat ground z={self.ground_z:.3f}"
        print(
            "[INFO] Height scan: "
            f"body={self.body_name}, points={self.dim}, offset={self.offset.tolist()}, "
            f"height_offset={self.height_offset:.3f}, geom_groups={self.geom_groups}, "
            f"backend={self.backend}, source={source}"
        )

    def scan(self, data) -> np.ndarray:
        if self.backend == "mj_ray":
            values, cache = mj_ray_height_scan(
                self.model,
                data,
                body_id=self.body_id,
                local_points=self.local_points,
                offset=self.offset,
                ground_z=self.ground_z,
                height_offset=self.height_offset,
                clip=self.clip,
                geom_groups=self.geom_groups,
            )
            self.last_scan = cache
            return values

        body_pos = np.asarray(data.xpos[self.body_id], dtype=np.float64)
        rot = _yaw_rotation_from_body_xmat(data.xmat[self.body_id])
        scan_origin = body_pos + rot @ self.offset
        points_w = scan_origin[None, :] + self.local_points @ rot.T
        heights = self._terrain_heights(points_w[:, :2], ray_z=float(scan_origin[2]))
        values = scan_origin[2] - self.offset[2] - heights - self.height_offset
        values = np.clip(values, self.clip[0], self.clip[1])
        values = values.reshape(1, -1).astype(np.float32)
        hit_points = np.column_stack((points_w[:, 0], points_w[:, 1], heights))
        self.last_scan = {
            "scan_origin": scan_origin.astype(np.float64),
            "ray_starts": points_w.astype(np.float64),
            "hit_points": hit_points.astype(np.float64),
            "values": values.copy(),
            "geom_ids": np.full(points_w.shape[0], -1, dtype=np.int32),
        }
        return values

    def _terrain_heights(self, xy: np.ndarray, *, ray_z: float) -> np.ndarray:
        if self.mesh is None:
            return np.full(xy.shape[0], self.ground_z, dtype=np.float64)

        zmax = float(np.asarray(self.mesh.bounds, dtype=np.float64)[1, 2])
        origins = np.column_stack((xy[:, 0], xy[:, 1], np.full(xy.shape[0], max(ray_z, zmax) + 2.0, dtype=np.float64)))
        directions = np.tile(np.asarray([0.0, 0.0, -1.0], dtype=np.float64), (xy.shape[0], 1))
        locations, index_ray, _ = self.mesh.ray.intersects_location(origins, directions, multiple_hits=True)
        heights = np.full(xy.shape[0], self.ground_z, dtype=np.float64)
        for ray_index in np.unique(index_ray):
            hit_z = locations[index_ray == ray_index, 2]
            heights[int(ray_index)] = float(np.max(hit_z))
        return heights

    def draw(
        self,
        viewer,
        *,
        points_only: bool = False,
        max_points: int | None = None,
        z_offset: float = 0.12,
        point_radius: float = 0.08,
    ) -> int:
        if self.last_scan is None:
            return 0

        import mujoco

        hit_points = np.asarray(self.last_scan["hit_points"], dtype=np.float64)
        ray_starts = np.asarray(self.last_scan["ray_starts"], dtype=np.float64)
        if hit_points.size == 0:
            return 0
        hit_points = hit_points.copy()
        hit_points[:, 2] += float(z_offset)
        if max_points is not None:
            hit_points = hit_points[: max(0, int(max_points))]
            ray_starts = ray_starts[: hit_points.shape[0]]

        point_rgba = np.asarray([0.0, 1.0, 0.15, 1.0], dtype=np.float32)
        ray_rgba = np.asarray([0.0, 1.0, 1.0, 0.65], dtype=np.float32)
        radius = max(0.001, float(point_radius))
        sphere_size = np.asarray([radius, radius, radius], dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        added = 0

        with viewer.lock():
            scn = viewer.user_scn
            max_geoms = getattr(scn, "maxgeom", None)
            if max_geoms is None:
                max_geoms = len(scn.geoms)
            max_geoms = int(max_geoms)
            for ray_start, hit_point in zip(ray_starts, hit_points):
                if not points_only and scn.ngeom < max_geoms:
                    geom = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3, dtype=np.float64),
                        np.zeros(3, dtype=np.float64),
                        identity_mat,
                        ray_rgba,
                    )
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        1.0,
                        np.asarray(ray_start, dtype=np.float64),
                        np.asarray(hit_point, dtype=np.float64),
                    )
                    geom.rgba[:] = ray_rgba
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                    scn.ngeom += 1
                    added += 1
                if scn.ngeom >= max_geoms:
                    break
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    sphere_size,
                    np.asarray(hit_point, dtype=np.float64),
                    identity_mat,
                    point_rgba,
                )
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                scn.ngeom += 1
                added += 1
        return added
