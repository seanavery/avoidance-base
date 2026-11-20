"""Frame-system chain and footprint tests."""

from __future__ import annotations

import importlib
import math
import sys

import numpy as np
import pytest
from viam.proto.common import Geometry, Pose, PoseInFrame, RectangularPrism, Transform, Vector3
from viam.proto.robot import FrameSystemConfig

from src.frames import (
    base_box_forward_lateral,
    orientation_vector_to_rotation_matrix,
    pose_matrix_in_destination,
    resolve_footprint,
)


def _part(name: str, parent: str, pose: Pose, box: tuple[float, float, float] | None = None):
    geometry = None
    if box is not None:
        geometry = Geometry(box=RectangularPrism(dims_mm=Vector3(x=box[0], y=box[1], z=box[2])))
    return FrameSystemConfig(
        frame=Transform(
            reference_frame=name,
            pose_in_observer_frame=PoseInFrame(reference_frame=parent, pose=pose),
            physical_object=geometry,
        )
    )


def _identity(x: float, y: float, z: float) -> Pose:
    return Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0)


def test_camera_parented_to_world_is_rejected():
    configs = [
        _part("cart", "world", _identity(0, 0, 0)),
        _part("cam", "world", _identity(0, 200, 300)),
    ]
    assert pose_matrix_in_destination(configs, "cam", "cart") is None


def test_camera_chain_must_reach_the_base():
    configs = [
        _part("cart", "world", _identity(0, 0, 0)),
        _part("mast", "world", _identity(0, 100, 0)),
        _part("cam", "mast", _identity(0, 50, 0)),
    ]
    assert pose_matrix_in_destination(configs, "cam", "cart") is None


def test_camera_mounted_through_a_link_on_the_base():
    configs = [
        _part("cart", "world", _identity(0, 0, 90), box=(400, 800, 200)),
        _part("mast", "cart", _identity(0, 100, 0)),
        _part("cam", "mast", _identity(10, 50, 200)),
    ]
    transform = pose_matrix_in_destination(configs, "cam", "cart")
    assert transform is not None
    assert transform[0, 3] == 10
    assert transform[1, 3] == 150
    assert transform[2, 3] == 200
    box = base_box_forward_lateral(configs, "cart")
    assert box == (0.8, 0.4)
    length, width = resolve_footprint(box, length_m=None, width_m=0.5, robot_radius_m=0.25)
    assert length == 0.8
    assert width == 0.5
    length, width = resolve_footprint(None, length_m=None, width_m=None, robot_radius_m=0.25)
    assert (length, width) == (0.5, 0.5)


def test_orbbec_forward_axis_is_base_forward():
    # Live cart: orbbec ov (0, 1, 0, -90). +Z is optical forward and must land on +Y.
    pose = Pose(x=0, y=230, z=310, o_x=0, o_y=1, o_z=0, theta=-90)
    configs = [
        _part("cart", "world", _identity(0, 0, 0)),
        _part("cam", "cart", pose),
    ]
    transform = pose_matrix_in_destination(configs, "cam", "cart")
    assert transform is not None
    point = transform[:3, :3] @ [0.0, 0.0, 1000.0] + transform[:3, 3]
    assert abs(point[0] - 0.0) < 1e-4
    assert abs(point[1] - 1230.0) < 1e-4
    assert abs(point[2] - 310.0) < 1e-4


def _rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_orientation_vector_known_rotations():
    np.testing.assert_allclose(orientation_vector_to_rotation_matrix(0, 0, 1, 0), np.eye(3), atol=1e-12)
    # A zero vector means +Z, as in RDK.
    np.testing.assert_allclose(orientation_vector_to_rotation_matrix(0, 0, 0, 0), np.eye(3), atol=1e-12)
    # Theta spins about the vector, in degrees.
    np.testing.assert_allclose(orientation_vector_to_rotation_matrix(0, 0, 1, 90), _rz(90), atol=1e-12)
    # +Z tipped onto +X is a 90 degree turn about +Y.
    np.testing.assert_allclose(
        orientation_vector_to_rotation_matrix(1, 0, 0, 0),
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        atol=1e-12,
    )
    # Camera-optical mount used on the live carts, ov (0, 1, 0, -90): optical +X right stays right,
    # +Y down stays down, +Z forward lands on base +Y forward.
    np.testing.assert_allclose(
        orientation_vector_to_rotation_matrix(0, 1, 0, -90),
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        atol=1e-12,
    )
    # South pole: +Z flipped, measured as if longitude were 0.
    np.testing.assert_allclose(
        orientation_vector_to_rotation_matrix(0, 0, -1, 0),
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        atol=1e-12,
    )


def test_orientation_vector_is_a_rotation_whose_z_axis_is_the_vector():
    rng = np.random.default_rng(7)
    for _ in range(200):
        vector = rng.normal(size=3)
        theta = float(rng.uniform(-360, 360))
        rotation = orientation_vector_to_rotation_matrix(*vector, theta)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        np.testing.assert_allclose(rotation[:, 2], vector / np.linalg.norm(vector), atol=1e-12)


def test_orientation_vector_is_continuous_through_the_poles():
    # Inside the pole radius theta is measured as if longitude were 0, so a vector a hair off +Z
    # rotates like +Z itself instead of jumping by its longitude.
    for theta in (0.0, 30.0, -120.0):
        for tiny in ((1e-6, 0.0), (0.0, 1e-6), (-1e-6, -1e-6)):
            np.testing.assert_allclose(
                orientation_vector_to_rotation_matrix(tiny[0], tiny[1], 1, theta), _rz(theta), atol=1e-5
            )
            np.testing.assert_allclose(
                orientation_vector_to_rotation_matrix(tiny[0], tiny[1], -1, theta),
                orientation_vector_to_rotation_matrix(0, 0, -1, theta),
                atol=1e-5,
            )


def test_frames_work_without_viam_spatialmath(monkeypatch):
    # viam.spatialmath only exists in viam-sdk releases that need Python 3.10+. On Python 3.8
    # (e.g. Jetson JetPack 5) pip installs viam-sdk 0.72, which doesn't have it.
    monkeypatch.setitem(sys.modules, "viam.spatialmath", None)
    frames = importlib.reload(importlib.import_module("src.frames"))
    configs = [
        _part("cart", "world", _identity(0, 0, 0)),
        _part("cam", "cart", Pose(x=0, y=230, z=310, o_x=0, o_y=1, o_z=0, theta=-90)),
    ]
    assert frames.pose_matrix_in_destination(configs, "cam", "cart") is not None


def test_orientation_vector_matches_viam_spatialmath_when_available():
    spatialmath = pytest.importorskip("viam.spatialmath")
    rng = np.random.default_rng(11)
    vectors = [rng.normal(size=3) for _ in range(300)]
    # Exact poles only: within ~1e-4 of a pole, current RDK keeps +Z on the vector while the SDK's
    # rust-utils still uses the older longitude-0 rule, so they differ by about the vector's offset.
    vectors += [np.array(v, dtype=float) for v in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (0, -1, 0))]
    for vector in vectors:
        theta = float(rng.uniform(-180, 180))
        ov = spatialmath.OrientationVector(*vector, math.radians(theta))
        expected = np.asarray(ov.to_quaternion().to_rotation_matrix().elements, dtype=float).reshape(3, 3)
        np.testing.assert_allclose(orientation_vector_to_rotation_matrix(*vector, theta), expected, atol=1e-9)
