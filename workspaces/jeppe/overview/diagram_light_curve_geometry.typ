#import "@preview/cetz:0.5.2": canvas, draw

#canvas({
  import draw: *

  // --- Asteroid body (irregular polygon) ---
  let pts = (
    (0.0,  0.0),
    (-0.6, 1.3),
    (-0.9, 2.6),
    (-0.5, 3.8),
    (0.5,  4.4),
    (1.7,  4.7),
    (2.8,  4.6),
    (4.0,  4.2),
    (4.7,  3.3),
    (5.0,  2.1),
    (4.7,  0.9),
    (3.8,  0.1),
    (2.7, -0.3),
    (1.3, -0.3),
  )
  line(..pts, close: true, fill: luma(92%), stroke: luma(30%) + 0.8pt)

  // --- Surface patch ds (ellipse at centre of asteroid) ---
  let patch = (2.5, 2.0)
  circle(patch, radius: (0.9, 0.45), stroke: rgb("#7c6fd4") + 1.2pt, fill: none,
    name: "patch")
  content((patch.at(0), patch.at(1) - 0.7),
    text(fill: rgb("#7c6fd4"), style: "italic", [d#text(style: "normal")[s]]))

  // --- Normal vector n-hat (straight up) ---
  let n-tip = (patch.at(0), patch.at(1) + 3.2)
  line(patch, n-tip, stroke: luma(20%) + 1.2pt,
    mark: (end: ">", fill: luma(20%), size: 0.22), name: "nvec")
  content((n-tip.at(0) + 0.25, n-tip.at(1) + 0.1),
    anchor: "west",
    $hat(n)$)

  // --- Observer vector E-hat (upper right) ---
  let obs = (6.8, 5.6)
  line(patch, obs, stroke: rgb("#1a6bb5") + 1.2pt,
    mark: (end: ">", fill: rgb("#1a6bb5"), size: 0.22))
  circle(obs, radius: 0.38, stroke: rgb("#1a6bb5") + 1pt, fill: white)
  content(obs, text(fill: rgb("#1a6bb5"), style: "italic", size: 8pt, $hat(E)$))
  content((obs.at(0) + 0.45, obs.at(1)),
    anchor: "west",
    text(fill: rgb("#1a6bb5"), size: 8pt, [(Observer)]))

  // --- Sun vector E0-hat (upper left) ---
  let sun = (-1.8, 5.6)
  line(patch, sun, stroke: rgb("#c07c10") + 1.2pt,
    mark: (end: ">", fill: rgb("#c07c10"), size: 0.22))
  circle(sun, radius: 0.38, stroke: rgb("#c07c10") + 1pt,
    fill: rgb("#fef3d0"))
  content(sun, text(fill: rgb("#c07c10"), size: 10pt, [☀]))
  content((sun.at(0), sun.at(1) - 0.65),
    text(fill: rgb("#c07c10"), style: "italic", size: 8pt, $hat(E)_0$))
  content((sun.at(0), sun.at(1) - 1.1),
    text(fill: rgb("#c07c10"), size: 8pt, [(Sun)]))

  // --- μ arc between n and E ---
  let r = 1.4
  arc(patch, start: 48deg, stop: 90deg, radius: r,
    stroke: rgb("#1a6bb5") + 0.7pt, fill: none)
  content((patch.at(0) + r * calc.cos(60deg) + 0.15,
           patch.at(1) + r * calc.sin(60deg)),
    anchor: "west",
    text(fill: rgb("#1a6bb5"), $mu$))

  // --- μ0 arc between n and E0 ---
  arc(patch, start: 90deg, stop: 132deg, radius: r,
    stroke: rgb("#c07c10") + 0.7pt, fill: none)
  content((patch.at(0) + r * calc.cos(120deg) - 0.15,
           patch.at(1) + r * calc.sin(120deg)),
    anchor: "east",
    text(fill: rgb("#c07c10"), $mu_0$))
})
