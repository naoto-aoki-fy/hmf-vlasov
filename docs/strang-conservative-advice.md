## Verdict

`strang_conservative` is **not properly implemented according to `strang-splitting-conservative-spec.md` as written**. The code has the right *high-level Strang-splitting structure* and a conservative-looking primitive-difference remap, but it violates key finite-volume grid assumptions in the spec. The largest defect is the momentum grid: `dv` is computed as if the stored values were nodal points, while the conservative remap treats them as finite-volume cell averages. This makes the `p`-advection operator inconsistent with the spec and even non-identity when the force is zero.

## What is implemented correctly

The selected scheme does route `scheme == "strang_conservative"` to `advance_x_conservative` and `advance_v_conservative`. The main loop also implements the usual optimized repeated-Strang pattern: an initial angular half-step, then repeated `B` and combined angular steps, ending with a final angular half-step. This is equivalent to repeated `A_{dt/2} B_dt A_{dt/2}` steps when adjacent angular half-steps are combined. ([GitHub][1])

The conservative remap functions also use the primitive-difference form required by the spec: evaluate a primitive at traced-back left and right cell edges, then divide the difference by the cell width. That matches the spec’s formula for semi-Lagrangian finite-volume advection. ([GitHub][2])

The force formula has the correct sign: the code computes `force = cos(x)*My - sin(x)*Mx`, which matches (F_i=-M_x\sin\theta_i+M_y\cos\theta_i). ([GitHub][2])

## Main noncompliance: the grid is not the specified finite-volume grid

The spec requires a **cell-centered finite-volume grid**:

[
\theta_i=(i+1/2)\Delta\theta,\qquad
p_j=-p_{\max}+(j+1/2)\Delta p,\qquad
\Delta p=\frac{2p_{\max}}{N_p}.
]

It also says the unknown is the **cell average** (\bar f_{i,j}). ([GitHub][2])

The code instead does this:

```python
self.dx = (self.xmax - self.xmin) / self.Nx
self.dv = (self.vmax - self.vmin) / (self.Nv - 1)
self.x = self.xmin + np.arange(self.Nx) * self.dx
self.v = self.vmin + np.arange(self.Nv) * self.dv
```

So `x` and `v` are endpoint/nodal-style coordinates, not cell centers; worse, `dv` is computed with `Nv - 1`, not `Nv`. ([GitHub][1])

For the finite-volume spec, this should be closer to:

```python
self.dx = (self.xmax - self.xmin) / self.Nx
self.dv = (self.vmax - self.vmin) / self.Nv
self.x = self.xmin + (np.arange(self.Nx) + 0.5) * self.dx
self.v = self.vmin + (np.arange(self.Nv) + 0.5) * self.dv
```

Using `[-π, π)` rather than `[0, 2π)` is fine by periodicity, but using left endpoints as cell centers is not.

## Serious defect in `advance_v_conservative`

The momentum remap builds edges as:

```python
edges = g.vmin + np.arange(g.Nv + 1) * g.dv
```

but `g.dv` was defined as `(vmax - vmin)/(Nv - 1)`. Therefore the final edge is not `vmax`; it is `vmax + dv`. The bounded primitive then clamps values at `x >= vmax` to the total mass. ([GitHub][1])

This breaks a basic invariant: if `force == 0`, the momentum advection step (B_{\Delta t}) should be the identity. In the current implementation, it is not. For example, with one row

```text
[1, 2, 3, 4, 5]
```

and zero force, the implemented remap sends it to

```text
[1, 2, 3, 9, 0]
```

using the code’s own primitive/clamping logic. Total mass is preserved in this toy case, but the distribution is not. That is incompatible with the split equation (\partial_t f + F_i\partial_p f=0) when (F_i=0), and therefore incompatible with the spec’s (p)-advection substep. ([GitHub][2])

This bug may be partly hidden when the last velocity column is initially zero and boundary mass stays negligible, but the spec explicitly treats the numerical unknowns as valid cell averages over the entire truncated momentum grid. A zero-force `B` step must not move the last cell’s mass into the penultimate cell.

## Angular remap is closer, but still grid-inconsistent

`advance_x_conservative` uses periodic primitive wrapping and computes:

```python
fnew[:, j] = (G(right) - G(left)) / dx
```

for each fixed velocity index. That is structurally consistent with the spec’s conservative angular advection formula, and it should conserve the mass of each fixed-index row. ([GitHub][2])

However, it uses `g.v[j]` as the advection velocity. Since `g.v` contains endpoint/nodal values instead of finite-volume cell-centered (p_j), the advection speeds are not the (p_j) specified by the scheme. The angular remap formula is mostly right; the data layout feeding it is not. ([GitHub][2])

## Reconstruction choice: acceptable in spirit, but fragile for waterbags

The primitive is built by cubic-spline interpolation of cumulative cell masses. This is a conservative primitive-difference reconstruction in the sense that the primitive passes through cumulative mass values at cell edges. The spec allows conservative cubic reconstruction as an option, though it notes WENO is better for sharp waterbag edges. ([GitHub][2])

But the implementation has no limiter or positivity preservation. For discontinuous waterbag data, cubic primitive reconstructions can overshoot and produce negative cell averages after remap. The spec does not absolutely require positivity preservation, but it does require tracking (f_{\min}) and boundary mass for accuracy assessment. ([GitHub][2])

## Diagnostics are incomplete

The spec says diagnostics should include mass, energy, magnetization, (L_2), positivity check (f_{\min}), and momentum-boundary mass (B_p). ([GitHub][2])

The code writes mass, energy, kinetic/internal energy, momentum, `Mx`, `My`, `I2`, and `I3`, but does not write `f_min` or momentum-boundary mass. `I2` is effectively the (L_2)-type integral, but the positivity and boundary-mass diagnostics are missing. ([GitHub][1])

## Minimal fixes

The most important correction is to make the stored arrays genuinely represent cell averages on the spec’s cell-centered grid:

```python
def __post_init__(self):
    self.dx = (self.xmax - self.xmin) / self.Nx
    self.dv = (self.vmax - self.vmin) / self.Nv

    self.x_edges = self.xmin + np.arange(self.Nx + 1) * self.dx
    self.v_edges = self.vmin + np.arange(self.Nv + 1) * self.dv

    self.x = self.x_edges[:-1] + 0.5 * self.dx
    self.v = self.v_edges[:-1] + 0.5 * self.dv

    self.f = np.zeros((self.Nx, self.Nv))
    self.rho = np.zeros(self.Nx)
    self.phi = np.zeros(self.Nv)
    self.force = np.zeros(self.Nx)
```

Then update the conservative remaps to use the stored edge arrays:

```python
def advance_x_conservative(g: Grid, h: float):
    tau = g.DT * h
    edges = g.x_edges
    fnew = np.empty_like(g.f)

    for j in range(g.Nv):
        shift = g.v[j] * tau
        left = edges[:-1] - shift
        right = edges[1:] - shift
        G_left = eval_periodic_primitive_from_cell_averages(
            g.f[:, j], g.dx, left - g.xmin
        )
        G_right = eval_periodic_primitive_from_cell_averages(
            g.f[:, j], g.dx, right - g.xmin
        )
        fnew[:, j] = (G_right - G_left) / g.dx

    g.f = fnew


def advance_v_conservative(g: Grid, h: float):
    tau = g.DT * h
    edges = g.v_edges
    fnew = np.empty_like(g.f)

    for i in range(g.Nx):
        shift = g.force[i] * tau
        left = edges[:-1] - shift
        right = edges[1:] - shift
        G_left = eval_bounded_primitive_from_cell_averages(
            g.f[i, :], g.dv, g.vmin, g.vmax, left
        )
        G_right = eval_bounded_primitive_from_cell_averages(
            g.f[i, :], g.dv, g.vmin, g.vmax, right
        )
        fnew[i, :] = (G_right - G_left) / g.dv

    g.f = fnew
```

Also add at least these invariant tests:

```python
def test_v_advection_zero_force_is_identity():
    g = Grid(Nx=4, Nv=8, xmin=-np.pi, xmax=np.pi, vmin=-2.5, vmax=2.5, DT=0.1)
    rng = np.random.default_rng(0)
    g.f = rng.random((g.Nx, g.Nv))
    g.force[:] = 0.0

    f0 = g.f.copy()
    advance_v_conservative(g, 1.0)

    np.testing.assert_allclose(g.f, f0, rtol=1e-13, atol=1e-13)


def test_x_advection_row_mass_conserved():
    g = Grid(Nx=16, Nv=5, xmin=-np.pi, xmax=np.pi, vmin=-2.5, vmax=2.5, DT=0.1)
    rng = np.random.default_rng(1)
    g.f = rng.random((g.Nx, g.Nv))

    row_mass0 = g.f.sum(axis=0) * g.dx
    advance_x_conservative(g, 0.5)
    row_mass1 = g.f.sum(axis=0) * g.dx

    np.testing.assert_allclose(row_mass1, row_mass0, rtol=1e-13, atol=1e-13)
```

## Bottom line

`strang_conservative` is **partially implemented**: the Strang composition, force timing, primitive-difference remap pattern, periodic angular wrapping, and bounded momentum primitive are present. But the implementation is **not faithful to the spec** because the grid is not cell-centered, `dv` is wrong for finite-volume cell averages, and the momentum remap can fail the zero-force identity property. Fixing the grid definition is mandatory before this can be considered a proper implementation of the documented conservative semi-Lagrangian Strang scheme.

[1]: https://github.com/naoto-aoki-fy/hmf-vlasov/blob/main/hmf_vlasov.py "hmf-vlasov/hmf_vlasov.py at main · naoto-aoki-fy/hmf-vlasov · GitHub"
[2]: https://github.com/naoto-aoki-fy/hmf-vlasov/blob/main/strang-splitting-conservative-spec.md "hmf-vlasov/strang-splitting-conservative-spec.md at main · naoto-aoki-fy/hmf-vlasov · GitHub"

