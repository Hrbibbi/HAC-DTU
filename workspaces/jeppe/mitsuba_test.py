import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import mitsuba as mi
    import numpy as np

    import matplotlib.pyplot as plt

    from hac.data import get_model_stl_path

    return get_model_stl_path, mi, np, plt


@app.cell
def _(get_model_stl_path):
    # Load
    model_stl_path = get_model_stl_path(3)
    return (model_stl_path,)


@app.cell
def _():
    # mi.set_variant("scalar_rgb")

    # # Check mesh bounds first
    # print(f"Mesh bounds: {m.bounds}")
    # print(f"Mesh centroid: {m.centroid}")
    # print(f"Mesh extents: {m.extents}")

    # # Auto-scale based on mesh size
    # mesh_size = max(m.extents)
    # camera_distance = mesh_size * 2.5

    # scale = 1.0
    # s = mi.load_dict(
    #     {
    #         "type": "scene",
    #         "integrator": {"type": "path"},
    #         "sensor": {
    #             "type": "perspective",
    #             "to_world": mi.ScalarTransform4f.look_at(
    #                 origin=[0.0, -camera_distance, camera_distance * 0.5],
    #                 target=[0.0, 0.0, 0.0],
    #                 up=[0.0, 0.0, 1.0],
    #             ),
    #             "fov": 35,
    #             "sampler": {
    #                 "type": "independent",
    #                 "sample_count": 32,
    #             },
    #             "film": {
    #                 "type": "hdrfilm",
    #                 "width": 512,
    #                 "height": 512,
    #                 "rfilter": {"type": "box"},
    #             },
    #         },
    #         "sun": {
    #             "type": "directional",
    #             "direction": [-1.0, -1.0, -1.0],
    #             "irradiance": {
    #                 "type": "rgb",
    #                 "value": 5.0,
    #             },
    #         },
    #         "asteroid": {
    #             "type": "obj",
    #             "filename": str(model_obj_path),
    #             "face_normals": False,
    #             "to_world": mi.ScalarTransform4f.scale([scale, scale, scale]),
    #             "bsdf": {
    #                 "type": "roughplastic",
    #                 "diffuse_reflectance": {
    #                     "type": "rgb",
    #                     "value": [0.45, 0.42, 0.38],
    #                 },
    #                 "alpha": 0.6,
    #             },
    #         },
    #     }
    # )

    # print("Rendering scene...")
    # img = mi.render(s)
    # print(f"Render complete. Image shape: {img.shape}")

    # fig = plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.imshow(mi.util.convert_to_bitmap(img))
    # plt.tight_layout()

    # fig  # Return the figure for Marimo to display
    return


@app.cell
def _(mi, model_stl_path, np):
    from hac.mitsuba import make_scene
    from hac.mitsuba import MitsubaMesh, RenderingSettings

    m = MitsubaMesh.from_stl_path(model_stl_path)

    # sun_direction = np.asarray([-1.0, -1.0, -1.0])
    camera_vector = np.asarray([0.0, -1.0, 0.5])
    camera_distance = 100.0

    width, height = 128, 128
    ssp = 32  # Number of samples per pixel

    rendering_settings = RenderingSettings(width=width, height=height, spp=32)

    theta = 180.0
    alpha = 0.0
    phi = 0.0
    observer_distance = 1.0

    mi.set_variant("scalar_rgb")  # Need to recompile mitsuba to get scalar_mono
    scene = make_scene(
        mesh=m,
        theta=theta,
        alpha=alpha,
        phi=phi,
        observer_distance=observer_distance,
        asteroid_scale=0.2,
        rendering_settings=rendering_settings,
        albedo=1.0,
        # sun_direction=sun_direction,
    )

    params = mi.traverse(scene)
    params
    return m, make_scene, observer_distance, rendering_settings, scene, theta


@app.cell
def _(mi, plt, scene):
    def render_scene(
        scene: mi.Scene,
    ):
        """
        Render one frame and return the bitmap before lightcurve integration.
        """
        return mi.render(scene)


    def display_rendering(img, ax=None):
        """
        Display the rendered image using matplotlib.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.axis("off")
        ax.imshow(mi.util.convert_to_bitmap(img))

        return ax


    img = render_scene(scene)
    ax = display_rendering(img)

    ax
    return (render_scene,)


@app.cell
def _(m, make_scene, np, observer_distance, rendering_settings, theta):
    # Render a little movie
    alpha_anim = 0.0
    N_PHI = 360
    phis_anim = np.linspace(0, 360, N_PHI)

    scenes = [
        make_scene(
            mesh=m,
            theta=theta,
            alpha=alpha_anim,
            phi=phi,
            observer_distance=observer_distance,
            asteroid_scale=0.2,
            rendering_settings=rendering_settings,
            albedo=1.0,
        )
        for phi in phis_anim
    ]
    return (scenes,)


@app.cell
def _(render_scene, scenes):
    # Render scenes
    renders = [render_scene(scene) for scene in scenes]
    return (renders,)


@app.cell
def _(mi, renders):
    bitmaps = [mi.util.convert_to_bitmap(img) for img in renders]
    return (bitmaps,)


@app.cell
def _(bitmaps):
    # display_rendering(renders[100])
    type(bitmaps[100])

    bitmaps[100].pixel_format()

    return


@app.cell
def _(bitmaps):
    bitmaps[100]
    return


@app.cell
def _(bitmaps, np, plt):
    a = np.asarray(bitmaps[100])
    plt.imshow(a)
    return


@app.cell
def _(mi, np, renders):
    from hac.lightcurve_integration import intensity

    lightcurve = [
        intensity(np.asarray(mi.util.convert_to_bitmap(render))) for render in renders
    ]
    return (lightcurve,)


@app.cell
def _(lightcurve, np):
    lightcurve_normalised = lightcurve / np.mean(lightcurve)
    return (lightcurve_normalised,)


@app.cell
def _(lightcurve_normalised, plt):
    from hac.data import load_data

    ds = load_data(model_num=3, origin="simulated", curve_type="intensity")
    ds = load_data(model_num=3, origin="simulated", curve_type="intensity")

    plt.plot(lightcurve_normalised, label="Mitsuba simulation")
    plt.plot(ds.data["0_hor1"], label="HAC simulation")
    return


@app.cell
def _(lightcurve, plt):
    plt.plot(lightcurve)
    return


if __name__ == "__main__":
    app.run()
