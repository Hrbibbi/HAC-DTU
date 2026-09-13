import sys
import trimesh
from IPython.display import Image, display

from hac.constants import MODEL_BASE_RADII
from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.convex_inversion_multi import ConvexInversionMulti, _facet_normals_and_areas
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python script.py <asteroid_number> <origin>")
    sys.exit(1)

A = int(sys.argv[1])      # asteroid number
origin = sys.argv[2]      # e.g. simulated, observed, ...

optimizer = "cg"

CR = MODEL_BASE_RADII[A]
CI = ConvexInversionMulti(cylinder_radius=CR, max_faces=1500)

CI.load_lightcurve_data(
    model_num=A,
    origin=origin,
    curve_type="intensity",
)

LSLparam = 0.8
CI.set_scattering_params(curve_type="intensity", value=LSLparam)

CI.build_forward()
print("total init areas", CI.areas_init_total)

compute = True
creg = 0.2

while compute:
    print(f"Computing asteroid {A} with regularization parameter lambda={creg}",flush=True)

    CI.fit_areas(
        convexity_regularization=creg,
        optimizer=optimizer,
        scale_penalty=None,
        max_iter=30000,
        avoid_gpu=False,
    )

    print("area sum", np.sum(CI.areas_opt))

    CI.refine_areas()

    if CI.dark_area_percentage > 1:
        creg *= 2
    else:
        try:
            CI.minkowski_reconstruction(max_iter=20000, epsilon=0.005)
        except Exception as e:
            print(f"Error occurred while reconstructing asteroid {A}: {e}")
            compute = False
            continue

        v = CI.recon_vertices

        zmin, zmax = np.min(v[:, 2]), np.max(v[:, 2])
        v *= 2 / (zmax - zmin)

        minkowski_radius = np.max(np.linalg.norm(v[:, :2], axis=1))
        v *= CR / minkowski_radius

        zmin, zmax = np.min(v[:, 2]), np.max(v[:, 2])
        v[:, 2] -= zmin
        v[:, 2] *= 2 / (zmax - zmin)
        v[:, 2] -= 1

        CI.recon_vertices = v

        CI.save_reconstruction(out_dir="stl/intensity", views=False)

        with open(
            f"stl/intensity/asteroid{A}_{origin}_intensity_info.txt",
            "w",
        ) as f:
            f.write(f"Dark area percentage: {CI.dark_area_percentage}\n")
            f.write(f"Regularization parameter: {creg}\n")
            f.write(f"Area penalty parameter: {creg}\n")
            f.write(
                f"Reconstruction radius (before cylinder normalization): "
                f"{minkowski_radius}\n"
            )
            f.write(f"True radius: {CR}\n")
            f.write(f"LSL scattering parameter: {LSLparam}\n")
            f.write(f"Optimizer: {optimizer}\n")

        compute = False