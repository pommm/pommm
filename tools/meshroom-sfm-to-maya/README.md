# meshroom-sfm-to-maya

Convert a Meshroom / AliceVision `camera.sfm` (the output of the
**StructureFromMotion** node) into a Maya ASCII (`.ma`) scene: one camera
per reconstructed photo, or a single animated camera for a tracked video
shot.

## 1. Producing `camera.sfm` with Meshroom

This script only handles the *conversion* step — you still need Meshroom
itself (a GUI/CLI photogrammetry app, not something that runs inside this
tool) to reconstruct the scene from your photos or video frames:

```bash
# GUI: File > New, drag in your images, press "Start"
# or headless, from the command line:
meshroom_batch --input /path/to/images --output /path/to/output
```

The graph runs `CameraInit -> FeatureExtraction -> ImageMatching ->
FeatureMatching -> StructureFromMotion -> ...`. Once `StructureFromMotion`
finishes, its output folder contains `cameras.sfm` (sometimes named
`sfm.json` depending on version) with the solved camera poses and
intrinsics. That file is the input to this script. If you're tracking a
video, extract frames first (e.g. with `ffmpeg`) and feed the frame
sequence to Meshroom so `frameId` is populated in the sfm data.

## 2. Converting to Maya

```bash
python3 sfm_to_maya.py camera.sfm scene.ma
```

Options:

| Flag | Default | Purpose |
|---|---|---|
| `--animated` | off (static) | Emit one keyframed camera across the sequence instead of a separate static camera per view. Use this for a tracked video shot; use static (default) for a photoset reconstruction. |
| `--up {y,z}` | `y` | World up-axis of the reconstruction. AliceVision doesn't enforce a convention, so world coordinates pass through unchanged by default. Pass `z` if your scene was aligned Z-up. |
| `--scale FACTOR` | `1.0` | Uniform scale applied to camera positions, e.g. to convert SfM's arbitrary reconstruction units to real-world Maya units once you know the scale (from a calibration marker/known distance). |
| `--near-clip`, `--far-clip` | `0.1`, `10000` | Camera clip planes. |
| `--fps` | `24` | Frame rate for `--animated` output. |
| `--start-frame` | `1` | First frame number for `--animated` output. |
| `--image-planes` | off | Attach an `imagePlane` referencing the source photo to each static camera (handy for visually checking the reconstruction). |

No third-party dependencies — pure Python 3 standard library, so it also
runs fine outside Maya (no `mayapy` required).

## Coordinate conversion notes

- AliceVision stores, per pose, a world→camera rotation matrix and a camera
  center such that `X_cam = R * (X_world - C)`, using the computer-vision
  camera convention (local `+X` right, `+Y` down, `+Z` forward into the
  scene). Maya cameras look down local `-Z` with `+Y` up. The script folds
  the fixed 180°-about-X change of basis between the two conventions
  directly into the Euler-angle derivation (see the comments above
  `matrix_to_maya_euler_xyz` in `sfm_to_maya.py`), so the output uses
  Maya's default `xyz` rotate order with no extra correction needed on your
  end.
- World axes (translation and the world basis the rotation is expressed in)
  are passed through unchanged unless you pass `--up z`.
- Reconstruction scale is arbitrary (SfM only recovers geometry up to an
  unknown scale factor) unless Meshroom's pipeline included a scale
  reference (e.g. a `SfMTransform` node driven by markers/GPS). Use
  `--scale` once you've measured that factor.

## Testing

A synthetic `camera.sfm` (identity pose + a known rotated pose) was used to
verify the transform math by hand — see the derivation comments in
`sfm_to_maya.py`. To sanity-check against your own data without opening
Maya, just eyeball the emitted `translate`/`rotate` values against the
`center`/`rotation` fields for a couple of views in your `camera.sfm`.
