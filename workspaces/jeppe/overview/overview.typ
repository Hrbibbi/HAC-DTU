#set heading(numbering: "1.1.a")
#set math.equation(numbering: "(1)")
#let vv = x => math.bold(math.upright(x))
#let vvh = x => math.hat(math.bold(math.upright(x)))
#let wider = h(3em)
#show link: set text(fill: blue)
#show raw: it => highlight(it, fill: gray.transparentize(50%), extent: 2pt, radius: 3pt)
#let atan2 = math.op("atan2")
#let atan = math.op("atan")
#set par(justify: true)
#show figure.caption: set text(size: 0.9em)
#show figure.caption: set par(leading: 0.6em)
#import "@preview/physica:0.9.8": super-T-as-transpose

#show: super-T-as-transpose


= Simple Lightcurve Forward Model <sec:simple_forward_model>

In the notation of Kaasalainen, @ref:kaasalainenOptimizationMethodsAsteroid2001a, the light curve contribution of an asteroid surface patch $dif s$ that is _visible_ (not occulted by other parts of the asteroid) and _illuminated_ by the Sun is given by:
$
  dif L = S(μ, μ_0) ϖ dif s
$ <eq:simple_lightcurve_contribution>
where:
- $L$ is the observed brightness (light curve).
- $S(μ, μ_0)$ and $ϖ$ arise from the scattering law, see @sec:scattering_laws.
- $ϖ$ is the _albedo_ of the surface patch, which is assumed to be constant across the patch.
- $μ = vv(E) ⋅ vv(n)$ is the projection between the surface unit normal, $vv(n)$ and the unit vector toward the observer (Earth), $vv(E)$.
- $μ_0 = vv(E_0) ⋅ vv(n)$ is the projection between the surface unit normal, $vv(n)$ and the unit vector toward the Sun, $vv(E_0)$.

For the convex case, we may satisfy the conditions of _visibility_ and _illumination_ by checking that $μ > 0$ and $μ_0 > 0$, respectively.

Note also that @eq:simple_lightcurve_contribution omits the factor $1 \/ r^2$ and assumes this may be absorbed into the implied normalisation of the light curve intensities, alongside other factors such as the incident light intensity.

Further, the model given by @eq:simple_lightcurve_contribution assumes that both the observer and the Sun are sufficiently far away to satisfy the _far-field approximation_.
For the Sun, this implies that the incident light upon the asteroid arrives as parallel rays, thus giving a constant incident light direction, $vv(E_0)$ across the asteroid. For the observer, this again implies that the light arrives as parallel rays, which is equivalent to the assumption that the asteroid is observed as an _orthographic projection_.

#figure(
  include "diagram_light_curve_geometry.typ",
  caption: [Geometry of light curve contribution.],
) <fig:light_curve_geometry>

== Scattering Laws <sec:scattering_laws>

Common scattering laws include @ref:kaasalainenOptimizationMethodsAsteroid2001a:
$
   S_"L" & = μ μ_0             && wider "Lambert" \
  S_"LS" & = S_"L" / (μ + μ_0) && wider "Lommel-Seeliger" \
$

= Challenge Setup <sec:challenge_setup>

We summarise the challenge setup as given by the organisers on #link("https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/", "the challenge website").

== Geometry <sec:geometry>
#figure(
  grid(
    columns: 2,
    image("geometry_top_view.png"), image("geometry_side_view.png"),
  ),
  caption: [Geometry setup of the asteroid, Sun, and observer (Earth).],
) <fig:geometry>

So the geometry is set up as:
- Cartesian coordinates $(x, y, z)$
- Measurement angle $θ$ is the direction of view in the $x y$-plane, as labelled in the top-view figure. Thus $θ = 0°$ means the camera looks along $+x$ and is placed on the $-x$ side of the asteroid.
- Elevation angle $α$ is the angle above the $x y$-plane. Positive $α$ places the camera at positive $z$.

Thus we have objects:
- Sun at $vv(r)_"sun" = (-∞, 0, 0)$ with direction toward the Sun $vvh(E)_"sun" = (-1, 0, 0)$. The light travels in the opposite direction, along $+x$.
- Observer at $vv(r)_"obs" = r_"obs" (-cos(θ) cos(α), -sin(θ) cos(α), sin(α))$ with direction toward the observer $vvh(E)_"obs" = (-cos(θ) cos(α), -sin(θ) cos(α), sin(α))$
  - The optical/view direction from the observer to the asteroid is the opposite vector, $(cos(θ) cos(α), sin(θ) cos(α), -sin(α))$.
  - Where $r_"obs"$ is assumed to be sufficiently large that the asteroid observed approximately as its orthographic projection.

The elevation angles $α$ are given by the organisers on the challenge website and are listed in

#figure(
  table(
    columns: 2,
    align: (right, right),
    table.header([*Measurement angle, $θ$*], [*Elevation angle, $α$*]),
    $0°$, $21°$,
    $45°$, $26°$,
    $90°$, $26°$,
    $135°$, $26°$,
    $225°$, $24°$,
    $270°$, $24°$,
    $315°$, $24°$,
  ),
  caption: [Cylinder radius $R$ for each asteroid model.],
) <tbl:elevation_angles>

== Mesh Normalisation <sec:mesh_normalisation>

To mitigate the non-uniqueness in the produced shapes which arises from the normalisation of the light curve intensities, we are given information about the _bounding cylinder_ of each asteroid model. In particular, we are told that each asteroid is tangent to the planes $z = ±1$ and the radius $r=R$, where $R$ is given for each model:
#figure(
  table(
    columns: 2,
    align: (left, center),
    table.header([*Asteroid*], [*$R$*]),
    "Model 1", $1.120$,
    "Model 2", $1.420$,
    "Model 3", $0.880$,
    "Model 4", $1.475$,
    "Model 5", $1.220$,
    "Model 6", $0.925$,
    "Model 7", $1.205$,
    "Model 8", $1.240$,
    "Model 9", $0.670$,
    "Model 10", $3.950$,
  ),
  caption: [Cylinder radius $R$ for each asteroid model.],
) <tbl:cylinder_radius>

=== Mesh Normalisation Procedure <sec:mesh_normalisation_procedure>
Let $vv(v)_i = (x_i, y_i, z_i)$, $i = 1, …, N$ denote the mesh vertices. The
procedure normalises the mesh into the challenge cylinder $D(0, R) × [-1, 1]$,
tangent to the planes $z = ±1$ with radius $r = R$ (see @tbl:cylinder_radius).

First, the $x y$-centre is taken as the midpoint of the $x y$ bounding box,
$
  x_c = 1/2 (min_i x_i + max_i x_i), quad
  y_c = 1/2 (min_i y_i + max_i y_i),
$
and the vertices are centred in $x y$:
$
  tilde(x)_i = x_i - x_c, quad
  tilde(y)_i = y_i - y_c.
$
The radial extent after centring is
$
  R_"max" = max_i sqrt(tilde(x)_i^2 + tilde(y)_i^2),
$
which fixes the $x y$-scale factor $s_"xy" = R \/ R_"max"$ so that the
normalised mesh has maximum radius exactly $R$.

Independently, the $z$-extent is normalised to span $[-1, 1]$. With
$
  z_"min" = min_i z_i, quad
  z_"max" = max_i z_i, quad
  z_c = 1/2 (z_"min" + z_"max"),
$
the $z$-scale factor is $s_z = 2 \/ (z_"max" - z_"min")$.

The normalised vertices are then
$
  vv(v)'_i = mat(s_"xy" tilde(x)_i; s_"xy" tilde(y)_i; s_z (z_i - z_c)).
$ <eq:mesh_normalisation>

By construction, $vv(v)'_i$ satisfies $max_i sqrt(x_i'^2 + y_i'^2) = R$ and
$min_i z_i' = -1$, $max_i z_i' = +1$.


= Inferring the Observer Position <sec:inferring_observer_position>

By inspecting frames from the Blender-simulated videos, we can clear see significant perspective distortion, as shown in @fig:model2_simulation_perspective and @fig:model2_real_perspective. For the Blender simulated data, the left and right edges are clearly not parallel as would be expected for an orthographic view, which suggests that the far-field observer assumption is severely violated. For the real data, however, the left and right edges are much closer to being parallel, leading us to conclude that the far-field assumption may be valid for the real data.

#grid(
  columns: 2,
  gutter: 1cm,
  [
    #figure(
      image("model2_0top_view_simulation_0sec.png", height: 8cm),
      caption: [
        First frame of the simulation video `top0.mp4` for model 2.
        The camera looks down at the model with an elevation of $α =21°$ (see @tbl:elevation_angles), though the top-face is not visible since it is not illuminated. Image the black area above the visible face to contain the top face of the model.
        For an orthographic camera, the left and right edges of the visible face would be perfectly parallel.
        Here they are clearly not, which suggests _significant_ perspective distortion.
      ],
    ) <fig:model2_simulation_perspective>
  ],
  [
    #figure(
      image("model2_0top_view_real_19sec.png", height: 8cm),
      caption: [
        An equivalent frame from the real video for model 2. The elevation angle is expected to be the same as in the simulation, but we observe a much smaller perspective distortion, which suggests that the camera is much further away. The orthographic assumption may be justified for the real video.
      ],
    ) <fig:model2_real_perspective>
  ],
)

== Camera Position Calibration <sec:camera_calibration>
The script `calibrate_camera_from_registrations.py` estimates a pinhole camera position and orientation
from manually identified registrations.
Each registration pairs a known 3D coordinate of a point on the normalised mesh (see @sec:mesh_normalisation) with its pixel position in a video frame.
The registration is manual and was carried out using #link("https://www.meshlab.net/", "MeshLab") and #link("https://www.gimp.org/", "GIMP"), with the pairs being recorded in `registrations.json5`.

While producing the registrations was somewhat laborious, the resulting camera calibration seems to be fairly accurate, as can be seen in @fig:model1_simulation_registration_debug and @fig:model3_simulation_registration_debug2, which show the registrations overlaid on the corresponding video frames.

#grid(
  columns: 2,
  gutter: 1cm,
  [
    #figure(
      image("model_1_simulation_0center_frame0_debug.png", height: 7.5cm),
      caption: [
        Image showing the registrations for model 1 overlaid the first frame of the simulation video `center0.mp4`.
      ],
    ) <fig:model1_simulation_registration_debug>
  ],
  [
    #figure(
      image("model_3_simulation_0center_frame0_debug.png", height: 7.5cm),
      caption: [
        Image showing the registrations for model 3 overlaid the first frame of the simulation video `center0.mp4`.
      ],
    ) <fig:model3_simulation_registration_debug2>
  ],
)

The obtained camera positions and orientations are summarised in @tbl:camera_position_estimates alongside the estimated standard deviation of the camera-object distance, $r_"obs"$.

#figure(
  table(
    columns: 7,
    align: (right, right, right, right, right, right, right),
    table.header(
      [*Asteroid*],
      [*Dataset*],
      [*View*],
      [*Frame*],
      [*Elevation, $α med [°]$,*],
      [*Azimuth, $θ med [°]$*],
      [*Distance, $r_"obs"$*],
    ),
    "Model 1", `simulated`, `0center`, $0$, $0.29$, $179.7$, $3.98±0.10$,
    "Model 2", `simulated`, `0bottom`, $0$, $-21.47$, $180.0$, $7.42±0.32$,
    "Model 2", `simulated`, `0top`, $0$, $21.47$, $180.0$, $7.41±0.32$,
    "Model 3", `simulated`, `0center`, $0$, $0.13$, $180.0$, $4.96±0.29$,
  ),
  caption: [Obtained camera positions and orientations. ],
) <tbl:camera_position_estimates>


Note that only `simulated` data is included, though the inference was also attempted for real videos. These were significantly harder to create registrations for due to the poor quality of the recorded videos. Further, the videos are 'trimmed', which likely invalidates the assumption that the principal point is at the image centre (see @sec:perspective_camera_fitting). Further, these were much closer to the orthographic case, which makes the inference problem ill-conditioned. Rather than try to work around this issue, we instead conclude that the real data is sufficiently far away that the orthographic assumption is valid.



=== Perspective Camera Fitting <sec:perspective_camera_fitting>

*Note*: This section is a largely LLM generated summary, see @sec:llm.

For correspondence $i$, let
$vv(P)_i = (X_i, Y_i, Z_i)$ be a known point on the model and let
$vv(p)_i = (u_i, v_i)$ be its measured pixel position. If $vv(o)$ is the
chosen object centre, the script first centres the world point,
$vv(q)_i = vv(P)_i - vv(o)$, and transforms it into camera coordinates:
$
  mat(x_i; y_i; z_i) = R vv(q)_i + vv(t).
$
Here $R in "SO"(3)$ is the world-to-camera rotation and $vv(t)$ is the
translation after centring the object. The lowercase $(x_i, y_i, z_i)$ are
Mitsuba sensor-local coordinates, not the world coordinates $(X_i, Y_i, Z_i)$;
world $+z$ remains the up direction. Mitsuba's sensor-local axes are $+x$ to
the left, $+y$ up, and $+z$ forward. Raster coordinates $u$ and $v$ increase
in the opposite directions to sensor-local $x$ and $y$, respectively, and a
visible point has $z_i > 0$.

The intrinsic model assumes zero skew and square pixels, and therefore uses one
focal length $f$ in pixel units. With principal point $(c_x, c_y)$, perspective
division gives the predicted pixel
$
  hat(u)_i = c_x - f x_i / z_i, quad
  hat(v)_i = c_y - f y_i / z_i.
$
By default the principal point is fixed at the image centre,
$(c_x, c_y) = ((W - 1) / 2, (H - 1) / 2)$; it can instead be supplied or fitted.

The camera is calibrated by minimising reprojection error:
$
  min_(R, vv(t), f, c_x, c_y)
  sum_i norm(mat(hat(u)_i - u_i; hat(v)_i - v_i))_2^2.
$
The principal-point variables are omitted when it is fixed. Rotation is
represented by a three-component rotation vector $vv(omega)$, with
$R = exp([vv(omega)]_times)$. The focal length is parameterised as
$f = exp(phi)$ to keep it positive. If
$r = max_i norm(vv(q)_i)_2$, the forward translation is parameterised as
$t_z = r + exp(delta)$. Since the forward component of any rotated
$vv(q)_i$ is at least $-r$, this guarantees $z_i > 0$ for every supplied
point. Thus the fit has seven unknowns---three for rotation, three for
translation, and one for focal length---or nine when the principal point is
also fitted. At least four or five correspondences, respectively, are needed
to provide enough scalar equations, although additional well-spread,
non-coplanar points make the solution considerably better constrained.

This is a nonlinear least-squares problem. To reduce the risk of accepting a
poor local minimum, the script starts the optimiser from cameras looking at the
object from many azimuths and elevations and from four different initial focal
lengths, then retains the solution with the smallest residual. The optional
robust mode replaces the squared loss on each residual component by a soft-L1
loss, reducing the influence of an incorrectly marked correspondence. After
fitting, the singular values of the residual Jacobian are checked to reject a
locally rank-deficient calibration in which pose and intrinsics cannot be
identified separately.

Finally, the fitted extrinsics are converted into a camera position. If
$vv(C)$ denotes that position, then
$R(vv(P) - vv(C)) = R(vv(P) - vv(o)) + vv(t)$, and hence
$
  vv(C) = vv(o) - R^T vv(t).
$

In this convention, $R$ maps world-coordinate direction vectors into Mitsuba
sensor-local coordinates, so $R^T$ is the rotation part of the sensor's
`to_world` transform. Equivalently, its `look_at` arguments are
$
  "origin" & = vv(C), \
  "target" & = vv(C) + R^T mat(0; 0; 1), \
      "up" & = R^T mat(0; 1; 0).
$

For $vv(Delta) = vv(C) - vv(o)$, the reported distance, elevation, and azimuth
are
$
      d & = norm(vv(Delta))_2, \
  alpha & = atan2(Delta_z, sqrt(Delta_x^2 + Delta_y^2)), \
   beta & = atan2(Delta_y, Delta_x).
$
The reported azimuth $beta$ is the bearing of the _camera position_. With the
view-direction convention in @sec:geometry, the measurement angle is therefore
$theta = beta - 180 degree$ modulo $360 degree$, while the elevation is
$alpha$. The fitted focal length is converted to fields of view using
$
  "FOV"_x = 2 atan(W / (2 f)), quad
  "FOV"_y = 2 atan(H / (2 f)).
$

// = A Better Forward Model <sec:better_forward_model>

// While the experimental setup for both the camera and simulated data does satisfy the far-field assumption for the Sun, we have found in @sec:inferring_observer_position that the far-field assumption for the observer only holds for the real camera, not the simulated data, which must instead be rendered with a perspective camera.

// If we let $𝒮_θ$ denote the asteroid surface at rotational phase $θ$ and consider a point
// $vv(x) ∈ 𝒮_θ$ with outward unit normal $vv(n)(vv(x))$ with $vv(c)$ being the
// camera centre, we find the direction from the surface point toward the camera is given by
// $
//   vvh(E)(vv(x))
//   = (vv(c) - vv(x)) / norm(vv(c) - vv(x))_2.
// $
// The direction toward the distant Sun remains the constant vector
// $vvh(E)_0$, as before. Define
// $
//     μ (vv(x)) & = [vv(n)(vv(x)) ⋅ vvh(E)(vv(x))]_+, \
//   μ_0 (vv(x)) & = [vv(n)(vv(x)) ⋅ vvh(E)_0]_+,
// $
// where $[a]_+ = max(a, 0)$.

// Let $χ_"sun" (vv(x))$ indicate that the ray from $vv(x)$ toward the Sun is
// unoccluded. Under direct illumination, the radiance leaving the surface toward
// the camera is
// $
//   L_"o" (vv(x), vvh(E))
//   = χ_"sun" (vv(x))
//   E_"sun"
//   f_"r" (vvh(E)_0, vvh(E); vv(x))
//   μ_0 (vv(x)),
// $ <eq:outgoing_radiance>
// where $E_"sun"$ is the incident solar irradiance and $f_"r"$ is the surface
// reflectance law. For a diffuse surface of albedo $ϖ$,
// $f_"r" = ϖ / π$.

// == Camera and Image Formation

// Let $𝒟$ denote the sensor domain. Each sensor position $vv(q) ∈ 𝒟$ defines a
// camera ray. Denote the first intersection of this ray with $𝒮_φ$ by
// $vv(x)_φ (vv(q))$. The rendered image is
// $
//   I_φ (vv(q)) =
//   cases(
//     L_"o" (vv(x)_φ (vv(q)), vvh(E)) & wide "ray intersects asteroid",
//     0 & wide "otherwise",
//   ).
// $ <eq:rendered_image>

// Selecting the first intersection accounts for visibility from the camera.
// Illumination visibility is determined separately by the shadow ray represented
// by $χ_"sun"$.

// The predicted light-curve intensity at phase $φ$ is the integrated image
// intensity:
// $
//   L(φ)
//   = integral_(𝒟) I_φ (vv(q)) dif vv(q)
//   ≈ sum_(p ∈ "pixels") I_φ [p].
// $ <eq:image_brightness>

// Because the supplied light curves contain only relative intensities, the
// predicted curve is divided by its phase average:
// $
//   tilde(L)(φ_i)
//   = L(φ_i)
//   / ((1 / N) sum_(j=1)^N L(φ_j)).
// $ <eq:normalised_brightness>

// == Relation to the Far-Field Model

// Equivalently, the image integral can be written as a surface integral:
// $
//   L(φ)
//   = integral_(𝒮_φ)
//   χ_"obs" χ_"sun"
//   E_"sun" f_"r" μ_0 J_"cam"
//   dif A,
// $
// where $χ_"obs"$ denotes visibility from the camera and $J_"cam"$ is the
// area factor induced by the camera projection.

// For an orthographic camera, $J_"cam" ∝ μ$. With Lambertian reflection, the
// integrand is therefore proportional to
// $
//   ϖ μ μ_0,
// $
// which recovers the simple Lambert scattering law in
// @sec:simple_forward_model. For a perspective camera, $J_"cam"$ additionally
// depends on the position and depth of the surface point. Consequently, different
// parts of the asteroid do not contribute as though they were observed under a
// single parallel projection.

= Rendering <sec:rendering>

Several algorithms exist, where it should be noted that we only require a relatively simple ray-tracing algorithm here due to the many simplifications that are possible here as opposed to the general case. For instance, we may assume monochromaticity and ignore atmospheric refraction if we assume the observer is unobstructed by an atmosphere. See also @sec:conclusions_2026_06_19.

A flexible renderer is the Mitsuba framework, which is a research-oriented renderer with inverse rendering capabilities.

Mitsuba-specific considerations are covered below in @sec:mitsuba

= Mitsuba <sec:mitsuba>
This section is a 'living document' trying to explain some of the considerations and details of the Mitsuba model.

== Variants
In Mitsuba variants describe the data type used to model the light transform, see https://mitsuba.readthedocs.io/en/latest/src/key_topics/variants.html.

We want to model just the _luminance_ (single-channel) light levels, and we require auto-diff support (`ad`) and either CUDA or LLVM support, which gives us variants:
- `cuda_ad_mono` (GPU)
- `llvm_ad_mono` (CPU)

== Scene <sec:scene>
We need to use a single scene to render everything, which is set up by referencing a single mesh instance at different orientations and translations along the $z$-axis.

This is set up in `hac.mitsuba.scene`, see @fig:batch_scene for a visualisation of the scene.

#figure(
  image("batch_scene.png", height: 6cm),
  caption: [Batch scene setup for Asteroid 3, see `mitsuba_z_stack_viewer.ipynb` for details.],
) <fig:batch_scene>

== Forward Model

Within the scene construct described in @sec:scene, we also include parameters of the forward model that are used to render the views that are later turned into light curves. An example of the rendered views prior to
reduction into light-curves is shown in @fig:rendered_views.

#figure(
  image("batch_scene_projections.png", height: 8cm),
  caption: [
    Subset of sensor renderings in a batch scene for asteroid 3 using Mitsuba forward model.
    Phases are $𝜙 ∈ {0°, 90°, 180°}$ across, respectively.
  ],
) <fig:rendered_views>

== Optimising Parameters of Forward Model

We can use the known geometry and given videos for the first 3 models to optimise the parameters of the forward model.

Unfortunately, the organisers of the challenge have used _perspective rendering for the videos instead of orthographic rendering_. This, combined with the fact that the distance between the asteroid and camera is fairly small#footnote[The distance appears to be approximately 7.3 units for model 2, where 2 is the height of the model as given by the challenge description.] and would violate any orthographic assumption, means that we need to produce a reasonable guess of the camera position (alongside the given orientations) in order to produce a reasonable forward model.
Intuitively, differences arising from the geometry of the object and aberrations stemming from the perspective projection are not easily (at all?) discernable from the perspective of inverse rendering.

Problematically, we believe that the organisers have not used a single, consistent combination of camera positions and object scalings for the videos, which makes it difficult, if not impossible, to produce a forward model that is consistent across all the models. Since the videos are not available for models 4 – 10, our only hope is that the distances and object scales are _roughly_ consistent across the models.

Even if we were able to incorporate the camera position into the optimisation, this is easily achieved with Mitsuba given the constraints described in:
- #link("https://github.com/mitsuba-renderer/mitsuba3/commit/132a3bce69f2ddbff3d434964efb4441a4d37a02")
- #link("https://github.com/mitsuba-renderer/mitsuba3/discussions/151#discussioncomment-7097927")
- #link("https://github.com/mitsuba-renderer/mitsuba3/issues/1439")

See #link("https://teams.microsoft.com/l/message/19:Ahm9wVNt8Tp4g3ZwMmA2b8xUaKykOoriOH_wpebIM-Q1@thread.tacv2/1785877381328?tenantId=f251f123-c9ce-448e-9277-34bb285911d9&groupId=0129ab18-d614-45eb-8a03-b368c443abaa&parentMessageId=1785877381328&teamName=Helsinki%20Asteroid%20Challenge%202026&channelName=General&createdTime=1785877381328", "discussion on Teams") for more details.

As a best-effort estimate, we can infer the camera position from the simulated videos of the first 3 models. According to the organisers documentation, care has been taken to make the distance and object scaling consistent between the blender simulated video and the real video. Since the videos are not angle synchronised and doing so would be a bit of effort, we only infer the camera positions from the Blender data.

== Update: Orthographic Assumption

The data has been updated to use orthographic rendering. In this case, the silhouette-based `direct_projective` integrator is equivalent to the `direct` integrator, but with extra steps (including invalidation of the cached GPU kernel!), so we can now use the `direct` integrator for the orthographic case, which is much faster.

= Other

This section contains bits and bobs that don't fit elsewhere in the document.

== Conclusions 2026-06-19 <sec:conclusions_2026_06_19>
- Standard fully-featured ray tracing algorithms are a poor fit, as we do not need things like:
  - Refraction
  - (Multiple scattering)
  - Polychromaticity

We can get away with simpler algorithms, in particular many papers seem to prefer doing the initial convex checks (visibility + illumination).

= LLM Disclosure <sec:llm>

The code for determining the camera position and orientation from the correspondances was generated with assistance of an LLM. Similarly, the summary of the perspective camera fitting in @sec:perspective_camera_fitting was produced with help from an LLM.

#pagebreak(weak: true)
#bibliography("references.bib")
