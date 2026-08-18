Order:
- Run `make_first_frames.sh`
- Run `uv run python normalize_known_meshes.py`
- Fill in `registrations.json`
- Run `uv run python calibrate_camera_from_registrations.py`

To find mesh coordinates:
- Use the normalised meshes in /workspaces/jeppe/camera_perspective_inference/meshes/
- Open in MeshLAB
- For "0center_frame0.png" copy the XML below and CTRL+V in MeshLab to set view
- Input in `registrations.json` the mesh coordinates and the pixel coordinates of the corresponding point in the image

Note: For model2 we can use LLM-made tool to grab corners, see `generate_registrations_from_png.py`

For `0center`:

```xml
<!DOCTYPE ViewState>
<!DOCTYPE ViewState>
<project>
 <VCGCamera TranslationVector="10 0 0 1"
            CameraType="0"
            PixelSizeMm="0.0369161 0.0369161"
            RotationMatrix="0 -1 0 0  0 0 1 0  -1 0 0 0  0 0 0 1"
            BinaryData="0"
            ViewportPx="3405 1994"
            LensDistortion="0 0"
            FocalMm="116.73158"
            CenterPx="1702 997"/>
 <ViewSettings TrackScale="0.5550287"
               FarPlane="9.0207529"
               NearPlane="0.55502903"/>
</project>
```