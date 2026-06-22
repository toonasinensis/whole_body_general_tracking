#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import numpy as np
import os
import re
import trimesh
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

WBT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = WBT_ROOT / "data" / "omniretarget" / "g1_terrain"
MOTION_RE = re.compile(r"^(?P<terrain_id>climb_\d+)_z_scale_(?P<z_scale>\d+(?:\.\d+)?)$")


def _natural_key(text: str) -> tuple:
    parts = re.findall(r"\d+|\D+", text)
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def _parse_xyz(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    parts = [float(x) for x in value.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got {value!r}")
    return parts[0], parts[1], parts[2]


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if origin is None:
        return transform
    xyz = _parse_xyz(origin.attrib.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_xyz(origin.attrib.get("rpy"), (0.0, 0.0, 0.0))
    transform = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], "sxyz")
    transform[:3, 3] = xyz
    return transform


def _iter_urdf_mesh_specs(urdf_path: Path) -> list[tuple[Path, tuple[float, float, float], np.ndarray]]:
    root = ET.parse(urdf_path).getroot()
    elements = root.findall(".//collision")
    if not elements:
        elements = root.findall(".//visual")

    specs = []
    for element in elements:
        mesh_element = element.find("./geometry/mesh")
        if mesh_element is None:
            continue
        filename = mesh_element.attrib.get("filename")
        if not filename:
            continue
        scale = _parse_xyz(mesh_element.attrib.get("scale"), (1.0, 1.0, 1.0))
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        specs.append((mesh_path, scale, _origin_transform(element.find("origin"))))
    if not specs:
        raise ValueError(f"{urdf_path}: no collision/visual mesh specs found")
    return specs


def _load_urdf_as_mesh(urdf_path: Path) -> trimesh.Trimesh:
    meshes = []
    for mesh_path, scale, transform in _iter_urdf_mesh_specs(urdf_path):
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        mesh = mesh.copy()
        mesh.apply_scale(scale)
        mesh.apply_transform(transform)
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    combined.remove_unreferenced_vertices()
    return combined


def _motion_members(zip_path: Path, include_regex: str | None) -> list[str]:
    compiled = re.compile(include_regex) if include_regex else None
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        for name in archive.namelist():
            path = Path(name)
            if path.parent.name != "robot-terrain" or path.suffix.lower() != ".npz":
                continue
            stem = path.stem
            if MOTION_RE.match(stem) is None:
                continue
            if compiled is not None and compiled.search(stem) is None:
                continue
            members.append(name)
    return sorted(members, key=lambda item: _natural_key(Path(item).stem))


def _write_raw_motion(
    archive: zipfile.ZipFile,
    member: str,
    output_path: Path,
    terrain_id: str,
    z_scale: str,
    terrain_stl_path: Path,
    force: bool,
) -> tuple[float, int, int, np.ndarray, np.ndarray]:
    if output_path.exists() and not force:
        with np.load(output_path, allow_pickle=False) as raw:
            qpos = np.asarray(raw["qpos"], dtype=np.float32)
            fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    else:
        with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as raw:
            qpos = np.asarray(raw["qpos"], dtype=np.float32)
            fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            qpos=qpos,
            fps=np.asarray([fps], dtype=np.float32),
            terrain_id=np.asarray(terrain_id),
            z_scale=np.asarray(z_scale),
            terrain_stl=np.asarray(os.path.relpath(terrain_stl_path, output_path.parent)),
            source_member=np.asarray(member),
        )
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{member}: expected qpos shape [T, 36], got {qpos.shape}")
    return fps, int(qpos.shape[0]), int(qpos.shape[1]), qpos[:, 4:7].min(axis=0), qpos[:, 4:7].max(axis=0)


def prepare_dataset(args: argparse.Namespace) -> list[dict]:
    if not args.omniretarget_root:
        raise ValueError("Set OMNIRETARGET_ROOT or pass --omniretarget_root.")
    omniretarget_root = Path(args.omniretarget_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    zip_path = omniretarget_root / "robot-terrain.zip"
    terrain_root = omniretarget_root / "models" / "terrain"
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    if not terrain_root.is_dir():
        raise FileNotFoundError(terrain_root)

    raw_dir = output_root / "raw_qpos"
    stl_dir = output_root / "terrains" / "stl"
    expected_tracking_dir = output_root / "tracking_npz_data"
    manifest_path = output_root / "pairs.jsonl"
    csv_path = output_root / "pairs.csv"
    dataset_txt_path = output_root / "tracking_dataset.txt"
    stl_dir.mkdir(parents=True, exist_ok=True)
    expected_tracking_dir.mkdir(parents=True, exist_ok=True)

    def rel_output(path: Path) -> str:
        return path.relative_to(output_root).as_posix()

    members = _motion_members(zip_path, args.include_regex)
    if args.max_motions is not None:
        members = members[: args.max_motions]
    if not members:
        raise RuntimeError(f"No robot-terrain motions matched in {zip_path}")

    records: list[dict] = []
    with zipfile.ZipFile(zip_path) as archive:
        for index, member in enumerate(members):
            stem = Path(member).stem
            match = MOTION_RE.match(stem)
            if match is None:
                continue
            terrain_id = match.group("terrain_id")
            z_scale = match.group("z_scale")
            terrain_urdf = terrain_root / terrain_id / f"multi_boxes_z_scale_{z_scale}.urdf"
            if not terrain_urdf.is_file():
                raise FileNotFoundError(f"{stem}: missing terrain URDF {terrain_urdf}")

            raw_motion_path = raw_dir / f"{stem}.npz"
            terrain_stl_path = stl_dir / f"{stem}.stl"
            if terrain_stl_path.exists() and not args.force:
                terrain_mesh = trimesh.load(terrain_stl_path, force="mesh", process=False)
            else:
                terrain_mesh = _load_urdf_as_mesh(terrain_urdf)
                terrain_stl_path.parent.mkdir(parents=True, exist_ok=True)
                terrain_mesh.export(terrain_stl_path)

            fps, frames, qpos_dim, root_min, root_max = _write_raw_motion(
                archive=archive,
                member=member,
                output_path=raw_motion_path,
                terrain_id=terrain_id,
                z_scale=z_scale,
                terrain_stl_path=terrain_stl_path,
                force=args.force,
            )
            bounds = np.asarray(terrain_mesh.bounds, dtype=np.float64)
            record = {
                "index": index,
                "name": stem,
                "terrain_id": terrain_id,
                "z_scale": float(z_scale),
                "source_member": member,
                "raw_motion": rel_output(raw_motion_path),
                "tracking_motion": rel_output(expected_tracking_dir / f"{stem}.npz"),
                "terrain_stl": rel_output(terrain_stl_path),
                "fps": fps,
                "frames": frames,
                "duration_s": float((frames - 1) / fps) if frames > 1 else 0.0,
                "qpos_dim": qpos_dim,
                "root_xyz_min": root_min.astype(float).tolist(),
                "root_xyz_max": root_max.astype(float).tolist(),
                "terrain_bounds_min": bounds[0].astype(float).tolist(),
                "terrain_bounds_max": bounds[1].astype(float).tolist(),
            }
            records.append(record)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    dataset_txt_path.write_text(
        "\n".join(Path(record["tracking_motion"]).name for record in records) + "\n",
        encoding="utf-8",
    )

    terrain_span = np.asarray([record["terrain_bounds_max"] for record in records]) - np.asarray(
        [record["terrain_bounds_min"] for record in records]
    )
    max_span = terrain_span.max(axis=0)
    grid_cols = int(math.ceil(math.sqrt(len(records))))
    grid_rows = int(math.ceil(len(records) / grid_cols))
    summary = {
        "count": len(records),
        "raw_motion_dir": rel_output(raw_dir),
        "tracking_motion_dir": rel_output(expected_tracking_dir),
        "terrain_stl_dir": rel_output(stl_dir),
        "manifest": rel_output(manifest_path),
        "csv": rel_output(csv_path),
        "dataset_txt": rel_output(dataset_txt_path),
        "suggested_terrain_grid": {
            "num_rows": grid_rows,
            "num_cols": grid_cols,
            "size": [float(max(4.0, max_span[0] + 2.0)), float(max(4.0, max_span[1] + 2.0))],
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OmniRetarget G1 robot-terrain qpos motions and paired STL terrain meshes."
    )
    parser.add_argument("--omniretarget_root", default=os.environ.get("OMNIRETARGET_ROOT"))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--include_regex",
        default=None,
        help="Optional regex matched against stems such as climb_00_z_scale_1.0.",
    )
    parser.add_argument("--max_motions", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate existing raw npz/STL outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = prepare_dataset(args)
    output_root = Path(args.output_root).expanduser().resolve()
    print(f"Prepared {len(records)} paired OmniRetarget G1 terrain motions under {output_root}")
    print(f"Manifest: {output_root / 'pairs.jsonl'}")
    print(f"Raw qpos: {output_root / 'raw_qpos'}")
    print(f"Terrain STL: {output_root / 'terrains' / 'stl'}")


if __name__ == "__main__":
    main()
