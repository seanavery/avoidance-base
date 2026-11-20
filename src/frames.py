"""Resolve camera poses and the base footprint from the Viam frame system.

Poses stay in Viam's base frame: millimetres, +Y forward, +X right, +Z up.
A camera is usable only when walking its parents reaches the configured base
before ``world``.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

WORLD_FRAME = "world"
FRAME_SYSTEM_TIMEOUT_S = 5.0


# Within this distance of +/-Z (after normalising), RDK measures theta as if
# the longitude were 0. Same constant as RDK's orientationVectorPoleRadius.
_ORIENTATION_VECTOR_POLE_RADIUS = 1e-4


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def orientation_vector_to_rotation_matrix(o_x: float, o_y: float, o_z: float, theta_deg: float) -> np.ndarray:
    """3×3 rotation for a Viam orientation vector, ``theta`` in degrees as in the proto.

    A port of RDK's ``OrientationVector.Quaternion`` (spatialmath/orientationVector.go):
    the vector is the child's +Z axis, and the rotation is ``Rz(lon) Ry(lat) Rz(theta)``
    with ``lat = acos(oz)`` and ``lon = atan2(oy, ox)``. Near a pole RDK measures theta
    as if lon were 0, so theta is shifted by lon there. A zero vector means +Z.

    This deliberately avoids ``viam.spatialmath``: that module only exists in
    viam-sdk releases that need Python 3.10+, and Python 3.8 machines (Jetson
    JetPack 5) get viam-sdk 0.72, which doesn't have it.
    """
    vector = np.array([float(o_x), float(o_y), float(o_z)], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        vector = np.array([0.0, 0.0, 1.0])
    else:
        vector /= norm
    ox, oy, oz = (float(v) for v in vector)
    pole = 1.0 - abs(oz) <= _ORIENTATION_VECTOR_POLE_RADIUS

    lat = math.acos(max(-1.0, min(1.0, oz)))
    lon = math.atan2(oy, ox)
    theta = math.radians(float(theta_deg))
    if pole:
        theta = theta - lon if oz > 0 else theta + lon
    return _rotation_z(lon) @ _rotation_y(lat) @ _rotation_z(theta)


def _pose_to_matrix(pose) -> np.ndarray:
    """4×4 pose of a child in its parent, translation in millimetres."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = orientation_vector_to_rotation_matrix(pose.o_x, pose.o_y, pose.o_z, pose.theta)
    transform[0, 3] = float(pose.x)
    transform[1, 3] = float(pose.y)
    transform[2, 3] = float(pose.z)
    return transform


def _frame_entries(configs: Sequence[Any]) -> dict[str, Any]:
    """Map frame name → frame entry (``Transform`` or the same shape)."""
    out: dict[str, Any] = {}
    for cfg in configs:
        frame = getattr(cfg, "frame", None) or cfg
        name = str(getattr(frame, "reference_frame", "") or "")
        if name:
            out[name] = frame
    return out


def pose_matrix_in_destination(
    configs: Sequence[Any],
    frame_name: str,
    destination: str,
) -> Optional[np.ndarray]:
    """``destination_T_frame`` (millimetres), or ``None`` when the chain misses.

    The walk follows each frame's parent. If ``world`` appears before
    ``destination``, the camera is not mounted on the base and the result is
    ``None``.
    """
    frames = _frame_entries(configs)
    if frame_name == destination:
        return np.eye(4, dtype=float)
    if frame_name not in frames:
        return None

    transform = np.eye(4, dtype=float)
    current = frame_name
    seen: set[str] = set()
    while current != destination:
        if current in seen or len(seen) > 64:
            return None
        seen.add(current)
        entry = frames.get(current)
        if entry is None:
            return None
        pose_in_parent = entry.pose_in_observer_frame
        parent = str(getattr(pose_in_parent, "reference_frame", "") or "")
        if not parent:
            return None
        if parent == WORLD_FRAME and destination != WORLD_FRAME:
            return None
        transform = _pose_to_matrix(pose_in_parent.pose) @ transform
        current = parent
    return transform


def base_box_forward_lateral(
    configs: Sequence[Any],
    base_name: str,
) -> Optional[Tuple[float, float]]:
    """``(length_m, width_m)`` of the base box.

    Viam base axes: +Y forward is length, +X right is width.
    """
    frames = _frame_entries(configs)
    entry = frames.get(base_name)
    if entry is None:
        return None
    geom = getattr(entry, "physical_object", None)
    if geom is None or not geom.ByteSize():
        return None
    if geom.WhichOneof("geometry_type") != "box":
        return None
    dims = geom.box.dims_mm
    lateral = abs(float(dims.x)) / 1000.0
    forward = abs(float(dims.y)) / 1000.0
    if lateral < 1e-4 or forward < 1e-4:
        return None
    return forward, lateral


def resolve_footprint(
    box: Optional[Tuple[float, float]],
    *,
    length_m: Optional[float],
    width_m: Optional[float],
    robot_radius_m: float,
) -> Tuple[float, float]:
    """Length (forward) and width (lateral) in metres.

    An explicit side overrides the base box. A side that is still unset uses
    ``2 * robot_radius_m``. With neither box nor overrides, both sides do.
    """
    fallback = 2.0 * float(robot_radius_m)
    length = length_m if length_m is not None and length_m > 0 else None
    width = width_m if width_m is not None and width_m > 0 else None
    if box is not None:
        if length is None:
            length = float(box[0])
        if width is None:
            width = float(box[1])
    if length is None:
        length = fallback
    if width is None:
        width = fallback
    return length, width


async def abandon_after(awaitable, timeout_s: float):
    """Await ``awaitable``, and stop waiting after ``timeout_s``.

    ``asyncio.wait_for`` stays stuck when the cancelled call never finishes.
    A camera or frame RPC in that state used to freeze cloud refresh, and
    every drive command then stayed at zero until the module process restarted.
    """
    task = asyncio.ensure_future(awaitable)
    _done, _pending = await asyncio.wait({task}, timeout=timeout_s)

    def _discard(done: asyncio.Task) -> None:
        if not done.cancelled():
            done.exception()

    if task not in _done:
        task.add_done_callback(_discard)
        task.cancel()
        raise TimeoutError(f"timed out after {timeout_s:.1f}s")
    return task.result()


async def fetch_frame_system_config(robot, *, timeout_s: float = FRAME_SYSTEM_TIMEOUT_S) -> list[Any]:
    """``robot.get_frame_system_config()`` on the caller's event loop."""
    configs = await abandon_after(robot.get_frame_system_config(), timeout_s)
    return list(configs)
