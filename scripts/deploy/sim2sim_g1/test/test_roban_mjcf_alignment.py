from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
URDF_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/assets/roban_s22/urdf/biped_s17_hands.urdf"
)
MJCF_PATH = ROOT / "source/whole_body_tracking/whole_body_tracking/assets/roban_s22/xml/biped_s17.xml"
ARM_LINKS = tuple(f"zarm_{side}{index}_link" for side in ("l", "r") for index in range(1, 6))


def _values(element: ET.Element, attribute: str) -> np.ndarray:
    return np.fromstring(element.attrib[attribute], sep=" ", dtype=np.float64)


def test_roban_mjcf_arm_tree_and_parameters_match_training_urdf() -> None:
    urdf = ET.parse(URDF_PATH).getroot()
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    urdf_links = {link.attrib["name"]: link for link in urdf.findall("link")}
    urdf_joints = {joint.find("child").attrib["link"]: joint for joint in urdf.findall("joint")}

    for link_name in ARM_LINKS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        parent_id = int(model.body_parentid[body_id])
        joint = urdf_joints[link_name]
        inertial = urdf_links[link_name].find("inertial")

        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) == joint.find("parent").attrib["link"]
        np.testing.assert_allclose(model.body_pos[body_id], _values(joint.find("origin"), "xyz"), atol=1e-8)
        np.testing.assert_allclose(
            model.body_mass[body_id],
            float(inertial.find("mass").attrib["value"]),
            atol=1e-10,
        )


def test_roban_mjcf_arm_zero_pose_matches_training_urdf_fk() -> None:
    urdf = ET.parse(URDF_PATH).getroot()
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    mujoco.mj_forward(model, data)

    transforms = {"base_link": np.eye(4)}
    pending = {
        joint.find("child").attrib["link"]: joint
        for joint in urdf.findall("joint")
        if joint.find("child").attrib["link"] in ARM_LINKS
    }
    while pending:
        for child, joint in list(pending.items()):
            parent = joint.find("parent").attrib["link"]
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            xyz = _values(origin, "xyz")
            roll, pitch, yaw = _values(origin, "rpy")
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            rotation = np.array(
                (
                    (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                    (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                    (-sp, cp * sr, cp * cr),
                )
            )
            local = np.eye(4)
            local[:3, :3] = rotation
            local[:3, 3] = xyz
            transforms[child] = transforms[parent] @ local
            del pending[child]

    for link_name in ARM_LINKS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        np.testing.assert_allclose(data.xpos[body_id], transforms[link_name][:3, 3], atol=2e-6)
        np.testing.assert_allclose(data.xmat[body_id].reshape(3, 3), transforms[link_name][:3, :3], atol=2e-6)
