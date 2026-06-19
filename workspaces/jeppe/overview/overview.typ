#set heading(numbering: "1.a")
#let vv = (x) => math.bold(math.upright(x))
#let wider = h(3em)

= Light Curve Contribution

Under the condition an asteroid surface patch $dif s$ is _visible_ (not occulted by other parts of the asteroid) and _illuminated_ by the Sun, it contributes to the observed light curve as:
$
  dif L = S(μ, μ_0) ϖ dif s
$
where:
- $L$ is the observed brightness (light curve).
- $S(μ, μ_0)$ and $ϖ$ arise from the scattering law, see @sec:scattering_laws.
- $μ = vv(E) ⋅ vv(n)$ is the projection between the surface unit normal, $vv(n)$ and the unit vector toward the observer (Earth), $vv(E)$.
- $μ_0 = vv(E_0) ⋅ vv(n)$ is the projection between the surface unit normal, $vv(n)$ and the unit vector toward the Sun, $vv(E_0)$.

#figure(
  include("diagram_light_curve_geometry.typ"),
  caption: [Geometry of light curve contribution.]
)

= Scattering Laws <sec:scattering_laws>

Common scattering laws include:
$
  S_"L" &= μ μ_0
  &&wider "Lambert"\
  S_"LS" &= S_"L" / (μ + μ_0)
  &&wider "Lommel-Seeliger"\

$

= Ray-Tracing <sec:ray_tracing>

Several algorithms exist, where it should be noted that we only require a relatively simple ray-tracing algorithm here due to the many simplifications that are possible here as opposed to the general case. For instance, we may assume monochromaticity and ignore atmospheric refraction if we assume the observer is unobstructed by an atmosphere.

The following algorithm has complexity $cal(O)(N^2)$ and can be used to determine _convex hull_.

== Conclusions 2026-06-19
- Standard fully-featured ray tracing algorithms are a poor fit, as we do not need things like:
  - Refraction
  - (Multiple scattering)
  - Polychromaticity

We can get away with simpler algorithms, in particular many papers seem to prefer doing the initial convex checks (visibility + illumination).