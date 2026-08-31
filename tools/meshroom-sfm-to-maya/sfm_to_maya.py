#!/usr/bin/env python3
"""Convert a Meshroom / AliceVision camera.sfm (StructureFromMotion output)
into a Maya ASCII (.ma) scene containing one camera per reconstructed view
(or, with --animated, a single animated camera across a frame sequence).

Usage:
    python3 sfm_to_maya.py camera.sfm output.ma
    python3 sfm_to_maya.py camera.sfm output.ma --animated --fps 24
    python3 sfm_to_maya.py camera.sfm output.ma --up z --scale 100 --image-planes

No third-party dependencies (stdlib only), so it can run outside Maya/mayapy.
"""

import argparse
import json
import math
import os
import re
import sys

MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# SfM parsing helpers
# ---------------------------------------------------------------------------

def _num(d, keys, default=None):
    """Look up the first present key in `keys` on dict `d` and cast to float.
    AliceVision sfm JSON frequently stores numbers as strings.
    """
    for k in keys:
        if k in d and d[k] not in (None, ""):
            v = d[k]
            if isinstance(v, (list, tuple)):
                continue
            return float(v)
    return default


def _vec(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return [float(x) for x in d[k]]
    return default


def load_sfm(path):
    with open(path, "r") as f:
        data = json.load(f)

    intrinsics_by_id = {}
    for intr in data.get("intrinsics", []):
        intrinsics_by_id[str(intr["intrinsicId"])] = intr

    poses_by_id = {}
    for p in data.get("poses", []):
        poses_by_id[str(p["poseId"])] = p

    views = []
    for v in data.get("views", []):
        pose_id = str(v.get("poseId"))
        intr_id = str(v.get("intrinsicId"))
        pose = poses_by_id.get(pose_id)
        intr = intrinsics_by_id.get(intr_id)
        if pose is None:
            # View wasn't successfully localized by the SfM solve; skip it.
            continue
        transform = pose["pose"]["transform"]
        rotation = _vec(transform, ["rotation"])
        center = _vec(transform, ["center"])
        if rotation is None or center is None or intr is None:
            continue

        width = _num(v, ["width"]) or _num(intr, ["width"])
        height = _num(v, ["height"]) or _num(intr, ["height"])
        views.append({
            "viewId": str(v.get("viewId")),
            "path": v.get("path", v.get("viewId", "view")),
            "width": width,
            "height": height,
            "frameId": v.get("frameId"),
            "rotation": rotation,
            "center": center,
            "intrinsic": intr,
        })
    return views


def intrinsic_to_camera_attrs(intr, width, height):
    """Derive Maya camera-shape attributes (mm/inches) from an AliceVision
    pinhole intrinsic block. Handles the couple of schema variants seen
    across AliceVision releases.
    """
    sensor_w_mm = _num(intr, ["sensorWidth"])
    sensor_h_mm = _num(intr, ["sensorHeight"])

    px_focal = _num(intr, ["pxFocalLength", "pxInitialFocalLength"])
    focal_mm = _num(intr, ["focalLength"])

    if sensor_w_mm is None:
        # No sensor DB match: fall back to a nominal 35mm-equivalent sensor
        # width so the script still produces a usable (if unscaled) camera.
        sensor_w_mm = 36.0
    if sensor_h_mm is None:
        sensor_h_mm = sensor_w_mm * (height / width) if width and height else sensor_w_mm

    if px_focal is None and focal_mm is not None and width:
        px_focal = focal_mm * width / sensor_w_mm
    if focal_mm is None and px_focal is not None and width:
        focal_mm = px_focal * sensor_w_mm / width
    if focal_mm is None:
        # Last resort default (roughly a 50mm-equivalent normal lens).
        focal_mm = 50.0

    pp = _vec(intr, ["principalPoint"], [0.0, 0.0])
    # principalPoint is stored as a pixel offset from the image center.
    px_to_mm_x = sensor_w_mm / width if width else 0.0
    px_to_mm_y = sensor_h_mm / height if height else 0.0
    offset_h_in = (pp[0] * px_to_mm_x) / MM_PER_INCH
    # Image Y grows downward, Maya's vertical film offset grows upward.
    offset_v_in = -(pp[1] * px_to_mm_y) / MM_PER_INCH

    return {
        "focalLength": focal_mm,
        "horizontalFilmAperture": sensor_w_mm / MM_PER_INCH,
        "verticalFilmAperture": sensor_h_mm / MM_PER_INCH,
        "horizontalFilmOffset": offset_h_in,
        "verticalFilmOffset": offset_v_in,
    }


# ---------------------------------------------------------------------------
# Coordinate-system math
#
# AliceVision stores, per pose: a world->camera rotation R (row-major, 9
# values) and camera center C (world space) such that X_cam = R * (X_world -
# C), using the computer-vision camera convention (local +X right, +Y down,
# +Z forward/into the scene).
#
# Maya cameras look down local -Z with +Y up, +X right. Converting between
# the two local conventions is a fixed 180-degree flip about local X (Y and
# Z negate, X unchanged) -- this holds regardless of the reconstructed pose.
#
# Folding that flip into the algebra and solving for Maya's default "xyz"
# rotate-order Euler angles (where, using Maya's row-vector convention, the
# local-to-parent matrix is M = Rx * Ry * Rz) yields the closed-form below,
# derived directly from R's entries.
# ---------------------------------------------------------------------------

def matrix_to_maya_euler_xyz(r):
    r0, r1, r2, r3, r4, r5, r6, r7, r8 = r

    sy = max(-1.0, min(1.0, -r2))
    cy = math.sqrt(max(0.0, 1.0 - sy * sy))

    if cy > 1e-6:
        rx = math.atan2(-r5, -r8)
        ry = math.asin(sy)
        rz = math.atan2(r1, r0)
    else:
        # Gimbal lock (camera looking straight up/down world Y after the
        # CV->Maya flip): rz is not observable, fold it into rx.
        ry = math.asin(sy)
        rx = math.atan2(r7, -r4)
        rz = 0.0

    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def apply_up_axis(vec3, up):
    if up == "y":
        return vec3
    x, y, z = vec3
    # Z-up (right-handed) -> Y-up (right-handed): (x, y, z) -> (x, z, -y)
    return [x, z, -y]


def apply_up_axis_to_rotation(r, up):
    """Re-express a world-to-camera rotation matrix's world axes under the
    same permutation used by apply_up_axis, without touching the local
    (camera-space) axes it maps *into*.
    """
    if up == "y":
        return r
    r0, r1, r2, r3, r4, r5, r6, r7, r8 = r
    # R's columns are world-space basis directions (since X_cam = R * X_world
    # is really R's rows dotted with world X/Y/Z basis vectors); permuting
    # world axes (x,y,z)->(x,z,-y) means column1 <-> column2 with a sign
    # flip, applied per matrix row.
    def perm_row(a, b, c):
        return [a, c, -b]
    new0 = perm_row(r0, r1, r2)
    new1 = perm_row(r3, r4, r5)
    new2 = perm_row(r6, r7, r8)
    return new0 + new1 + new2


# ---------------------------------------------------------------------------
# Maya ASCII writing
# ---------------------------------------------------------------------------

_INVALID_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def sanitize_name(name):
    base = os.path.splitext(os.path.basename(name))[0]
    base = _INVALID_NAME_RE.sub("_", base)
    if not base or base[0].isdigit():
        base = "cam_" + base
    return base


def esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def write_header(f):
    f.write('//Maya ASCII scene\n')
    f.write('//Generated by meshroom-sfm-to-maya (sfm_to_maya.py)\n')
    f.write('requires maya "2018";\n')
    f.write('currentUnit -l centimeter -a degree -t film;\n')
    f.write('fileInfo "application" "meshroom_sfm_to_maya";\n')


def write_static_camera(f, name, tx, ty, tz, rx, ry, rz, attrs, image_path,
                         near, far, image_plane):
    f.write('createNode transform -n "%s";\n' % name)
    f.write('\tsetAttr ".translate" -type "double3" %.9g %.9g %.9g;\n' % (tx, ty, tz))
    f.write('\tsetAttr ".rotate" -type "double3" %.9g %.9g %.9g;\n' % (rx, ry, rz))
    f.write('createNode camera -n "%sShape" -p "%s";\n' % (name, name))
    f.write('\tsetAttr -k off ".v";\n')
    f.write('\tsetAttr ".rnd" no;\n')
    f.write('\tsetAttr ".ncp" %.9g;\n' % near)
    f.write('\tsetAttr ".fcp" %.9g;\n' % far)
    f.write('\tsetAttr ".ff" 0;\n')
    f.write('\tsetAttr ".cameraScale" 1;\n')
    f.write('\tsetAttr ".focalLength" %.9g;\n' % attrs["focalLength"])
    f.write('\tsetAttr ".horizontalFilmAperture" %.9g;\n' % attrs["horizontalFilmAperture"])
    f.write('\tsetAttr ".verticalFilmAperture" %.9g;\n' % attrs["verticalFilmAperture"])
    f.write('\tsetAttr ".horizontalFilmOffset" %.9g;\n' % attrs["horizontalFilmOffset"])
    f.write('\tsetAttr ".verticalFilmOffset" %.9g;\n' % attrs["verticalFilmOffset"])
    if image_plane and image_path:
        plane_name = name + "ImagePlane"
        f.write('createNode imagePlane -n "%s" -p "%sShape";\n' % (plane_name, name))
        f.write('\tsetAttr ".imageName" -type "string" "%s";\n' % esc(image_path))
        f.write('\tsetAttr ".displayOnlyIfCurrent" 1;\n')
        f.write('\tsetAttr ".depth" %.9g;\n' % (near * 10.0))


def write_anim_curve(f, node_name, target_attr_plug, kind, frame_values):
    f.write('createNode %s -n "%s";\n' % (kind, node_name))
    f.write('\tsetAttr ".tan" 18;\n')
    f.write('\tsetAttr -s %d ".ktv[0:%d]"' % (len(frame_values), len(frame_values) - 1))
    for frame, value in frame_values:
        f.write(' %.9g %.9g' % (frame, value))
    f.write(';\n')
    f.write('connectAttr "%s.output" "%s";\n' % (node_name, target_attr_plug))


def write_animated_camera(f, name, samples, attrs, near, far):
    """samples: list of (frame, tx, ty, tz, rx, ry, rz) sorted by frame."""
    f.write('createNode transform -n "%s";\n' % name)
    f.write('createNode camera -n "%sShape" -p "%s";\n' % (name, name))
    f.write('\tsetAttr -k off ".v";\n')
    f.write('\tsetAttr ".rnd" yes;\n')
    f.write('\tsetAttr ".ncp" %.9g;\n' % near)
    f.write('\tsetAttr ".fcp" %.9g;\n' % far)
    f.write('\tsetAttr ".focalLength" %.9g;\n' % attrs["focalLength"])
    f.write('\tsetAttr ".horizontalFilmAperture" %.9g;\n' % attrs["horizontalFilmAperture"])
    f.write('\tsetAttr ".verticalFilmAperture" %.9g;\n' % attrs["verticalFilmAperture"])
    f.write('\tsetAttr ".horizontalFilmOffset" %.9g;\n' % attrs["horizontalFilmOffset"])
    f.write('\tsetAttr ".verticalFilmOffset" %.9g;\n' % attrs["verticalFilmOffset"])

    channels = [
        ("translateX", "animCurveTL", 1),
        ("translateY", "animCurveTL", 2),
        ("translateZ", "animCurveTL", 3),
        ("rotateX", "animCurveTA", 4),
        ("rotateY", "animCurveTA", 5),
        ("rotateZ", "animCurveTA", 6),
    ]
    for attr, kind, idx in channels:
        node_name = "%s_%s" % (name, attr)
        values = [(s[0], s[idx]) for s in samples]
        write_anim_curve(f, node_name, '%s.%s' % (name, attr), kind, values)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def frame_number(view):
    if view.get("frameId") not in (None, "", "-1"):
        try:
            return int(view["frameId"])
        except (TypeError, ValueError):
            pass
    m = re.search(r"(\d+)(?=\.[A-Za-z0-9]+$)", os.path.basename(view["path"]))
    if m:
        return int(m.group(1))
    return None


def convert(sfm_path, out_path, mode, up, scale, near, far, fps, start_frame,
            image_plane):
    views = load_sfm(sfm_path)
    if not views:
        raise SystemExit("No localized views found in %s (empty or unsolved SfM)." % sfm_path)

    per_view = []
    for v in views:
        r = apply_up_axis_to_rotation(v["rotation"], up)
        c = apply_up_axis(v["center"], up)
        c = [x * scale for x in c]
        rx, ry, rz = matrix_to_maya_euler_xyz(r)
        attrs = intrinsic_to_camera_attrs(v["intrinsic"], v["width"], v["height"])
        per_view.append({
            "name": sanitize_name(v["path"]),
            "path": v["path"],
            "frame": frame_number(v),
            "t": c,
            "rot": (rx, ry, rz),
            "attrs": attrs,
        })

    with open(out_path, "w") as f:
        write_header(f)

        if mode == "static":
            used_names = set()
            for pv in sorted(per_view, key=lambda p: (p["frame"] is None, p["frame"], p["name"])):
                name = pv["name"]
                n = name
                i = 2
                while n in used_names:
                    n = "%s%d" % (name, i)
                    i += 1
                used_names.add(n)
                tx, ty, tz = pv["t"]
                rx, ry, rz = pv["rot"]
                write_static_camera(f, n, tx, ty, tz, rx, ry, rz, pv["attrs"],
                                     pv["path"], near, far, image_plane)
        else:
            ordered = sorted(per_view, key=lambda p: (p["frame"] is None, p["frame"]))
            if any(p["frame"] is None for p in ordered):
                for i, p in enumerate(ordered):
                    p["frame"] = i
            samples = []
            for i, p in enumerate(ordered):
                frame = start_frame + (p["frame"] - ordered[0]["frame"])
                tx, ty, tz = p["t"]
                rx, ry, rz = p["rot"]
                samples.append((frame, tx, ty, tz, rx, ry, rz))
            f.write('currentUnit -l centimeter -a degree -t "%dfps";\n' % fps)
            write_animated_camera(f, "sfmCamera", samples, ordered[0]["attrs"], near, far)

    return len(per_view)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sfm", help="Path to Meshroom/AliceVision camera.sfm (StructureFromMotion output)")
    ap.add_argument("output", help="Path to write the Maya ASCII (.ma) file")
    ap.add_argument("--animated", dest="mode", action="store_const", const="animated",
                     default="static",
                     help="Emit a single keyframed camera across the view sequence "
                          "(use when camera.sfm comes from a tracked video shot) "
                          "instead of one static camera per view (default).")
    ap.add_argument("--up", choices=["y", "z"], default="y",
                     help="World up-axis of the SfM reconstruction. AliceVision does not "
                          "enforce a convention; pass 'z' if your scene was Z-up "
                          "(e.g. aligned to a Z-up ground plane) to convert to Maya's Y-up. "
                          "Default 'y' passes world coordinates through unchanged.")
    ap.add_argument("--scale", type=float, default=1.0,
                     help="Uniform scale factor applied to camera positions, e.g. to convert "
                          "the SfM's arbitrary reconstruction units to real-world Maya units "
                          "once you know the scale (from a calibration object/marker).")
    ap.add_argument("--near-clip", type=float, default=0.1, dest="near")
    ap.add_argument("--far-clip", type=float, default=10000.0, dest="far")
    ap.add_argument("--fps", type=float, default=24.0,
                     help="Frame rate used for --animated output.")
    ap.add_argument("--start-frame", type=int, default=1,
                     help="Frame number of the first view in --animated output.")
    ap.add_argument("--image-planes", dest="image_plane", action="store_true",
                     help="Attach an imagePlane referencing the source photo to each "
                          "static camera (ignored in --animated mode).")
    args = ap.parse_args()

    count = convert(args.sfm, args.output, args.mode, args.up, args.scale,
                     args.near, args.far, args.fps, args.start_frame, args.image_plane)
    print("Wrote %d camera(s) from %s to %s (mode=%s, up=%s)" %
          (count, args.sfm, args.output, args.mode, args.up))


if __name__ == "__main__":
    main()
