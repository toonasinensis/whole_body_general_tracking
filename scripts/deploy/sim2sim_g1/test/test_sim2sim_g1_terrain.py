from __future__ import annotations

import numpy as np
import sys
import xml.etree.ElementTree as ET

from sim2sim_g1.terrain import mj_ray_height_scan, mujoco_xml_with_terrain_mesh, terrain_boxes_from_stl


class Data:
    def __init__(self, *, yaw: float = 0.0, z: float = 0.9):
        c = np.cos(yaw)
        s = np.sin(yaw)
        self.xpos = np.asarray([[0.0, 0.0, z]], dtype=np.float64)
        self.xmat = np.asarray([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float64)


def _with_fake_mujoco(monkeypatch, ray_fn):
    fake_mujoco = type("FakeMujoco", (), {"mj_ray": staticmethod(ray_fn)})
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)


def test_mj_ray_height_scan_casts_downward_rays_and_returns_flat_values(monkeypatch) -> None:
    calls = []

    def fake_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid):
        calls.append((pnt.copy(), vec.copy(), geomgroup, flg_static, bodyexclude))
        geomid[0] = 7
        return float(pnt[2])

    _with_fake_mujoco(monkeypatch, fake_ray)
    values, cache = mj_ray_height_scan(
        model=object(),
        data=Data(),
        body_id=0,
        local_points=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64),
        offset=np.asarray([0.0, 0.0, 20.0], dtype=np.float64),
    )

    assert values.shape == (1, 2)
    assert np.allclose(values, np.asarray([[0.4, 0.4]], dtype=np.float32))
    assert len(calls) == 2
    assert all(np.array_equal(call[1], np.asarray([0.0, 0.0, -1.0], dtype=np.float64)) for call in calls)
    assert all(call[3] == 1 for call in calls)
    assert all(call[4] == 0 for call in calls)
    assert np.allclose(cache["hit_points"][:, 2], 0.0)
    assert np.array_equal(cache["geom_ids"], np.asarray([7, 7], dtype=np.int32))


def test_mj_ray_height_scan_uses_ground_z_when_ray_misses(monkeypatch) -> None:
    def fake_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid):
        return -1.0

    _with_fake_mujoco(monkeypatch, fake_ray)
    values, cache = mj_ray_height_scan(
        model=object(),
        data=Data(z=1.0),
        body_id=0,
        local_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        offset=np.asarray([0.0, 0.0, 20.0], dtype=np.float64),
        ground_z=0.25,
    )

    assert np.allclose(values, np.asarray([[0.25]], dtype=np.float32))
    assert np.allclose(cache["hit_points"], np.asarray([[0.0, 0.0, 0.25]], dtype=np.float64))
    assert np.array_equal(cache["geom_ids"], np.asarray([-1], dtype=np.int32))


def test_mj_ray_height_scan_yaw_aligns_local_grid(monkeypatch) -> None:
    starts = []

    def fake_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid):
        starts.append(pnt.copy())
        return float(pnt[2])

    _with_fake_mujoco(monkeypatch, fake_ray)
    mj_ray_height_scan(
        model=object(),
        data=Data(yaw=np.pi * 0.5),
        body_id=0,
        local_points=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        offset=np.asarray([0.0, 0.0, 20.0], dtype=np.float64),
    )

    assert np.allclose(starts[0][:2], np.asarray([0.0, 1.0], dtype=np.float64), atol=1e-7)


def test_mujoco_xml_with_terrain_mesh_injects_mesh_and_preserves_relative_meshdir(tmp_path) -> None:
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    terrain = tmp_path / "terrain.stl"
    terrain.write_text("solid terrain\nendsolid terrain\n")
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text("""
<mujoco model="test">
  <compiler meshdir="meshes"/>
  <asset><mesh name="robot" file="robot.stl"/></asset>
  <worldbody><geom name="floor" type="plane"/></worldbody>
</mujoco>
""".strip())

    out_path, tmpdir, injected = mujoco_xml_with_terrain_mesh(
        xml_path, terrain, mode="mesh", collision_backend="mesh", geom_group=4
    )
    try:
        root = ET.parse(out_path).getroot()
        compiler = root.find("compiler")
        terrain_mesh = root.findall("asset")[-1].find("mesh[@name='wbc_terrain_mesh']")
        terrain_geom = root.findall("worldbody")[-1].find("geom[@name='wbc_terrain']")

        assert injected == terrain.resolve()
        assert compiler is not None
        assert compiler.get("meshdir") == str(mesh_dir.resolve())
        assert terrain_mesh is not None
        assert terrain_mesh.get("file") == str(terrain.resolve())
        assert terrain_geom is not None
        assert terrain_geom.get("type") == "mesh"
        assert terrain_geom.get("group") == "4"
        assert terrain_geom.get("contype") == "1"
        assert terrain_geom.get("conaffinity") == "1"
    finally:
        assert tmpdir is not None
        tmpdir.cleanup()


def test_mujoco_xml_with_terrain_boxes_adds_visual_mesh_and_collision_boxes(tmp_path) -> None:
    import trimesh

    terrain = tmp_path / "box.stl"
    trimesh.creation.box(extents=(2.0, 1.0, 0.4), transform=np.eye(4)).export(terrain)
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text("<mujoco><asset/><worldbody/></mujoco>")

    boxes = terrain_boxes_from_stl(terrain)
    assert len(boxes) == 1
    assert np.allclose(boxes[0]["pos"], (0.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(sorted(boxes[0]["size"][:2]), [0.5, 1.0], atol=1e-6)
    assert np.isclose(boxes[0]["size"][2], 0.2, atol=1e-6)

    out_path, tmpdir, _ = mujoco_xml_with_terrain_mesh(
        xml_path, terrain, mode="mesh", collision_backend="boxes", geom_group=4
    )
    try:
        root = ET.parse(out_path).getroot()
        visual = root.find(".//geom[@name='wbc_terrain_visual']")
        collision = root.find(".//geom[@name='wbc_terrain_box_000']")

        assert visual is not None
        assert visual.get("type") == "mesh"
        assert visual.get("contype") == "0"
        assert visual.get("conaffinity") == "0"
        assert collision is not None
        assert collision.get("type") == "box"
        assert collision.get("group") == "4"
        assert collision.get("contype") == "1"
        assert collision.get("conaffinity") == "1"
    finally:
        assert tmpdir is not None
        tmpdir.cleanup()
