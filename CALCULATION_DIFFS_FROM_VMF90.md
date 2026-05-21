# Calculation differences vs `pdebuyl/vmf90`

Compared files:
- Local: `hmf_vlasov.py`
- Upstream reference: `vmf90/src/Vlasov_module.f90`, `vmf90/src/HMF_module.f90`, `vmf90/src/vmf90_hmf.f90`

## 1) Spatial domain is halved
- Local grid is initialized with `x in [0, π)`:
  - `Grid(nx, nv, 0.0, PI, -vmax, vmax, dt)`
- vmf90 HMF waterbag is defined around `x=0` with condition `abs(x) <= width`, and in standard HMF runs the physical periodic domain is `[-π, π)`.
- Effect: all x-integrals (magnetization, potential energy, etc.) are computed over half the expected domain unless compensated elsewhere.

## 2) Waterbag initialization is not the same support as vmf90
- Local uses a wrapped mask:
  - `abs(((x + π) % (2π)) - π) <= width`
  - This is a periodic interval selection in a `2π` sense.
- With local `x ∈ [0, π)`, this mask behaves differently than vmf90’s direct `abs(x) <= width` over symmetric x-domain.
- vmf90 also supports optional perturbation `1 + ε cos(x)` and momentum shift `p0`; local port omits both.

## 3) Advection interpolation method differs (major numerical change)
- Local code uses **piecewise-linear interpolation** in both x and v advections.
- vmf90 uses **cubic splines**:
  - periodic spline in x (`spline_periodic`)
  - natural spline in v (`spline_natural`)
- Effect: different numerical diffusion/dispersion and long-time conservation behavior.

## 4) Time of diagnostics differs for field data writes
- vmf90 writes fields only when image cadence condition is met; observables are still written every `t_top`.
- Local currently writes fields (`f`, `rho`, `phi`) on every `top` step because `write(top)` is called in both branches.
- Effect: larger output and potentially different sampled field timeline vs vmf90 defaults.

## 5) Magnetization/force formula parity: same algebraic form (for non-external HMF)
- Local force: `force = cos(x) * My - sin(x) * Mx`
- vmf90 non-external HMF uses the same expression.
- However, due to domain/support/interpolation differences above, computed values still diverge in practice.

## 6) Missing vmf90 extended-HMF terms
- vmf90 has optional external-field/epsilon variant:
  - `force = cos(x)*(epsilon*My) - sin(x)*(epsilon*Mx + Hfield)`
  - and matching interaction energy correction.
- Local implementation has only the basic non-external model.

## 7) Split-step sequence mostly matches vmf90’s Cheng–Knorr structure
- Pattern is equivalent (`x(1/2)`, repeated `v(1), x(1)`, then `v(1), x(1/2)`).
- Main difference is still interpolation order and domain/init choices, not operator ordering.
