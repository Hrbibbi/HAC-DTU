from hac.simulate import generate_challenge_lightcurves
from hac.constants import *

model_num = 3
for cam_idx in POSITION_IDS:
    generate_challenge_lightcurves(
        model_num=model_num,
        cam_idx=cam_idx,
        sim_root="workspaces/david/out/sim_lc"
    )