#import "@preview/touying:0.7.4": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

#set text(size: 20pt)
#show figure.caption: set text(size: 16pt)
#show raw: it => highlight(it, extent: 3pt, fill: gray.transparentize(70%), radius: 5pt)
#show link: set text(font: "DejaVu Sans Mono", size: 0.8em, fill: navy.lighten(20%))

= Jeppe Update — 2026-07-28

#figure(
  grid(columns: 3,
  ..range(1, 3, inclusive: true).map(i => image("asteroid6_big" + str(i) + ".png")),
  ),
  caption: [Ugly yet motivational asteroid 6 reconstruction.]
) <fig:front-fig>

== Mitsuba Forward
- Code from last time cleaned up a bit and available in `hac.mitsuba`
  - Still a little messy
  - I may break quite a few things in the future, but usable by others for experiments for now.
- Overview of new modules:
  - `scene`: module for constructing batch Mitsuba scenes and rendering them
    - `debug_scene`: to make the nice plots shown later
  - `visualize`: module for visualizing meshes and light curves
  - `forward`: module for forward rendering of meshes and light curves
    - Differentiable, alledgedly also _PyTorch_ compatible

#cols(columns: (1fr, 1fr))[
== Batch Scenes — $z$-stack
- To (efficiently) render + store gradients of mesh at multiple angles and _phases_ (of the rotation), we set up a single scene
  - Each _phase_ is an affine transformation on reference to the mesh
  - i.e. stacked and rotated along $z$
  - Each of these has sensors for enabled observation angles
- *Live Demo?*

][
  #figure(
    image("batch_scene.png"),
    caption: [Batch scene setup for asteroid 3, see `mitsuba_z_stack_viewer.ipynb` for details.]
  )
]

#cols(columns: (1fr, 1fr))[
== Batch Scenes — Forward
- Forward rendering is then just rendering each sensor individually
- Then applying appropriate reduction to get light curve
- Note: Currently this is just sum without thresholding
  - Does not match _Otsu/binary_ or _intensity_ methods
  - Given no 'stray light', so maybe thresholding not needed to match _intensity_ method well?
  - Tricky to get thresholding to be differentiable, but possible with _sigmoid_ or other
][
#figure(
  image("batch_scene_projections.png", height: 85%),
  caption: [
    Subset of sensor renderings in a batch scene for asteroid 3 using Mitsuba forward model.\
    Phases are #box[$ϕ ∈ {0°, 90°, 180°}$] across, respectively.
  ]
)
]

== Optimization
- Follows #link("https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html")
- See `workspaces/jeppe/mitsuba_shape_optimization.py` for details
- Still lots to do, very basic at the moment
- Scale penalty is kind of wrong
  - Maybe only penalize (relative) point mass outside bounding cylinder?
- Try on all models
- New 'built-in' penalty by differentiably applying normalization to mesh as part of forward
  - Seems to have helped with 'pan-caking'

== Optimization (Currently)
#cols(columns: (1fr, 1fr))[
- Currently _relatively_ fast
  - A10 GPU: ≈ 30 mins
  - A100 GPU: ≈ 20 mins
  - DrJIT needs Compute Capability 7.5+, so no V100
  - Priority access to A10/A40/A100 would be useful later
][
- Parameters currently:
  - Phases: $N_ϕ = 180$
  - Faces: $N_f = 2500$
  - Samples per pixel: $N_"spp" = 8$
  - Resolution: $(H, W) = (64, 64)$
  - Iterations: $N_"iter" = 300$
  - Adam with LR $α = 2×10^(-2)$
  - "Large Steps" parameter, $λ = 25.0$
]

== Reconstruction: Asteroid 1
#cols(columns: (1fr, 1fr))[
- Loss diverges quickly
- Likely due to imperfect forward model and excellent starting guess
][
#image("asteroid1_loss.png", height: 40%)
]
#figure(
  image("asteroid1_comparison.png", height: 40%)
)

== Reconstruction: Asteroid 2
- Honestly pretty good given naïve starting guess
- Improvements to consider:
  - Better forward model
  - Better starting guess (convex hull?)
  - Procedural mesh refinement
#figure(
  image("asteroid2_comparison.png", height: 40%)
)

== Reconstruction: Asteroid 3
- Introduces very slight non-convexity
- May need to improve optimization routine to explore non-convexity better
- Tilt is strangely to the wrong side?
  - Optimizer may have gotten stuck in local minimum
#figure(
  image("asteroid3_comparison.png", height: 40%)
)

== Reconstruction: Asteroid 4
- Successfully (?) rotates initial guess
  - A little hard to see in @fig:asteroid4-comparison — easier in movie, *Live Demo?*
- We should explore different seeds and optimizations to see if we have any confidence in produced shape
- Actually quite nice lightcurve fit and loss, *Live Demo?*
- Optimize initial guess rapidly by optimizing for rotation of initial cylinder(s) only?
- Optimize initial guess by generating random meshes and picking the best one?
#figure(
  image("asteroid4_comparison.png", height: 35%)
) <fig:asteroid4-comparison>

== Reconstruction: Asteroid 5
- At least has not exploded
- Reconstructions are a little 'spiky', maybe need procedural mesh normalization/refinement?
#v(1fr)
#figure(
  image("asteroid5_comparison.png", height: 65%)
) <fig:asteroid5-comparison>

== Reconstruction: Asteroid 6
#cols(columns: (1fr, 1fr))[
- Has a very flat face
- Again drawing many initial guesses from a prior distribution of archetypes may really help
  - Or if we can good convex hull?
- Quite good lightcurves!
#v(1fr)
#figure(
  image("asteroid6_comparison.png", height: 50%)
) <fig:asteroid6-comparison>
][
#figure(
  image("asteroid6_lightcurve.png", height: 80%)
)
]

== Reconstruction: Asteroid 7
- I bet this has a hole it in or something awful like that
- We need initial guesses with different topologies, or some occasional refinement step that can make holes
- Very difficult since it will be highly non-differentiable to poke a hole in a mesh!
#v(1fr)
#figure(
  image("asteroid7_comparison.png", height: 60%)
) <fig:asteroid7-comparison>

== Reconstruction: Asteroid 8
#cols(columns: (1fr, 1fr))[
- Very poor reconstruction
- From lightcurve: clearly extreme structure
  - Exploit lightcurve periodicity to make better starting guess
  - Can generally inform starting guess by lightcurve amplitudes and periodicities?
#v(1fr)
#figure(
  image("asteroid8_comparison.png")
) <fig:asteroid8-comparison>
][
#figure(
  image("asteroid8_lightcurve.png")
)
]


== Reconstruction: Asteroid 9-10
- Still running when I finished up this presentation, but probably done now!
- Lets look! *Live Demo?*

== Lightcurves
#cols(columns: (1fr, 1fr))[
- Lightcurves tend to fit best for observation angles $θ ∈ {0°, 45°, 270°, 315°}$, less so for $θ ∈ {135°, 225°}$
- Should be improved with better forward model
  - Better scattering model
  - Better summation/reduction method? (thresholding?)
][
#figure(
  image("asteroid3_lightcurve.png", height: 70%),
  caption: [Lightcurve comparison for asteroid 3.]
)
]

== Summary/Coming up
- Try to improve forward model parameters by comparing known model meshes to their forward projections
- Make `ForwardModel` take a reduction method instead of hardcoding sum for lightcurve
- Maybe see if we can get speed-up with `u16` or `u8` instead of `u32`/`f32`?
- Making an `optimize` module under `hac.mitsuba` for common optimization routines
- Procedural mesh refinement?
- Should use all phases (360 for Blender, 842 for camera)
  - In current setup problem is actually over-determined, IIRC
- Should use camera data instead of only Blender data
- Speed up optimization (maybe some DrJit hackery? JIT Caching?)
- Asked HAC organisers whether Blender data has been corrected or not
  - No response yet

== Final remarks
- This method probable needs a good starting guess to work well
- Can we use Mitsuba forward with a convex inversion method?
- Would be keen to meet up with people in DTU Library / 307 for a half-day?
- Mainly note to self:
  - Start putting renderings, logs, etc in `/work3/projects/HAC-DTU/workspaces/<USER>`
  - Currently can't be seen by others
  - Most of these don't belong in `git`
  - Output from reconstructions shown can be found there (if I remembered to do it?)
