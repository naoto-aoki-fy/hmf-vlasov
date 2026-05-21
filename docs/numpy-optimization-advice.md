## Summary

The Python `for`-loops that are most suitable for NumPy/SciPy batching are the slice-wise interpolation/remap loops in `hmf_vlasov.py`:

| Function                 |           Current loop | Can remove? | Rewrite strategy                                                                                           |
| ------------------------ | ---------------------: | ----------: | ---------------------------------------------------------------------------------------------------------- |
| `advance_x`              | `for j in range(g.Nv)` |         Yes | Build one batched periodic `CubicSpline` over `x`, then gather each column’s matching spline coefficients. |
| `advance_v`              | `for i in range(g.Nx)` |         Yes | Build one batched natural `CubicSpline` over `v`, then gather each row’s matching spline coefficients.     |
| `advance_x_conservative` | `for j in range(g.Nv)` |         Yes | Batch the periodic primitive construction for all velocity columns.                                        |
| `advance_v_conservative` | `for i in range(g.Nx)` |         Yes | Batch the bounded primitive construction for all spatial rows.                                             |

The highest-priority targets are `advance_x_conservative` and `advance_v_conservative`, because the default config uses `Nx = 256`, `Nv = 512`, and `scheme = "strang_conservative"`. 

Do **not** try to vectorize the outer time-stepping loops in `run`: each step mutates `g.f`, recomputes density/magnetization/force, and depends on the previous step’s state. 

---

## Important caveat

`CubicSpline` already accepts multidimensional `y` through its `axis` argument, and stores coefficients in `cs.c` with shape `(4, n-1, …)`. ([docs.scipy.org][1])

However, a naïve rewrite like this is **not** the right solution:

```python
spline = CubicSpline(x_periodic, y_periodic, axis=0)
fnew = spline(x_eval)
```

For this repository, each velocity column or spatial row is evaluated at its **own** backtraced coordinates. A direct batched spline call tends to produce a cross-product-like output with an extra batch dimension. The better approach is:

1. Build one batched `CubicSpline`.
2. Use `cs.c` and `np.searchsorted`.
3. Gather the coefficient for the matching batch component only.

---

# 1. Add this shared helper

This helper evaluates a batch of independent cubic splines whose interpolation axis is axis `0`, with one trailing batch dimension.

```python
def eval_cubic_components_axis0(cs: CubicSpline, xq: np.ndarray, fill_value=None) -> np.ndarray:
    """
    Evaluate independent CubicSpline components without a Python loop.

    Parameters
    ----------
    cs:
        CubicSpline built with axis=0 and y.shape == (n_points, n_batch).
    xq:
        Query points with shape (n_query, n_batch).
        xq[:, j] is evaluated against spline component j.
    fill_value:
        If not None, points outside cs.x are filled with this value.

    Returns
    -------
    out:
        Array with shape (n_query, n_batch).
    """
    xp = cs.x
    nseg = xp.size - 1

    if fill_value is None:
        valid = np.ones_like(xq, dtype=bool)
        xwork = xq
    else:
        valid = (xq >= xp[0]) & (xq <= xp[-1])
        xwork = np.clip(xq, xp[0], xp[-1])

    seg = np.searchsorted(xp, xwork, side="right") - 1
    seg = np.clip(seg, 0, nseg - 1)

    t = xwork - xp[seg]
    batch = np.arange(xq.shape[1])[None, :]

    c = cs.c
    out = ((c[0, seg, batch] * t + c[1, seg, batch]) * t + c[2, seg, batch]) * t + c[3, seg, batch]

    if fill_value is not None:
        out = np.where(valid, out, fill_value)

    return out
```

---

# 2. Rewrite `advance_x`

Current code loops over each velocity index `j`, creates a periodic spline for `g.f[:, j]`, and evaluates it at `x_eval[:, j]`. 

Replace it with:

```python
def advance_x(g: Grid, h: float):
    x_back = g.x[:, None] - g.DT * h * g.v[None, :]
    period = g.xmax - g.xmin
    x_eval = ((x_back - g.xmin) % period) + g.xmin

    x_periodic = np.concatenate((g.x, [g.xmax]))
    y_periodic = np.vstack((g.f, g.f[:1, :]))

    spline = CubicSpline(
        x_periodic,
        y_periodic,
        axis=0,
        bc_type="periodic",
        extrapolate="periodic",
    )

    g.f = eval_cubic_components_axis0(spline, x_eval)
```

This removes:

```python
for j in range(g.Nv):
    ...
```

---

# 3. Rewrite `advance_v`

Current code loops over each spatial index `i`, creates a natural spline for `g.f[i, :]`, evaluates it at `v_back[i, :]`, and replaces out-of-domain values with zero. 

Replace it with:

```python
def advance_v(g: Grid, h: float):
    v_back = g.v[None, :] - g.DT * h * g.force[:, None]

    spline = CubicSpline(
        g.v,
        g.f.T,
        axis=0,
        bc_type="natural",
        extrapolate=False,
    )

    g.f = eval_cubic_components_axis0(
        spline,
        v_back.T,
        fill_value=0.0,
    ).T
```

The transpose is intentional:

* `g.f.T` has shape `(Nv, Nx)`.
* Each spatial row becomes one batched spline component.
* `v_back.T` has shape `(Nv, Nx)`.
* The result is transposed back to `(Nx, Nv)`.

---

# 4. Batch the periodic primitive for `advance_x_conservative`

The current conservative `x` remap loops over `j`, builds the periodic primitive from `g.f[:, j]`, and evaluates left/right cell edges.  

Add this batched primitive helper:

```python
def eval_periodic_primitive_batch_axis0(cell_avg: np.ndarray, dx: float, x: np.ndarray) -> np.ndarray:
    """
    Batched version of eval_periodic_primitive_from_cell_averages.

    Parameters
    ----------
    cell_avg:
        Shape (n_cells, n_batch). For this code: (Nx, Nv).
    dx:
        Cell width.
    x:
        Query points relative to the periodic domain start.
        Shape (n_query, n_batch).

    Returns
    -------
    Primitive values with shape (n_query, n_batch).
    """
    n, n_batch = cell_avg.shape
    length = n * dx

    masses = cell_avg * dx
    p_edges = np.vstack(
        (
            np.zeros((1, n_batch), dtype=cell_avg.dtype),
            np.cumsum(masses, axis=0),
        )
    )

    total = p_edges[-1, :]
    x_edges = np.arange(n + 1) * dx

    q_edges = p_edges - (x_edges[:, None] / length) * total[None, :]

    q_spline = CubicSpline(
        x_edges,
        q_edges,
        axis=0,
        bc_type="periodic",
        extrapolate="periodic",
    )

    wraps = np.floor(x / length)
    x_mod = x - wraps * length

    q_val = eval_cubic_components_axis0(q_spline, x_mod)

    return wraps * total[None, :] + q_val + (x_mod / length) * total[None, :]
```

Then rewrite `advance_x_conservative` as:

```python
def advance_x_conservative(g: Grid, h: float):
    tau = g.DT * h

    edges_rel = np.arange(g.Nx + 1) * g.dx
    shift = g.v[None, :] * tau

    left = edges_rel[:-1, None] - shift
    right = edges_rel[1:, None] - shift

    g_left = eval_periodic_primitive_batch_axis0(g.f, g.dx, left)
    g_right = eval_periodic_primitive_batch_axis0(g.f, g.dx, right)

    g.f = (g_right - g_left) / g.dx
```

This removes the loop over `Nv`.

---

# 5. Batch the bounded primitive for `advance_v_conservative`

The current conservative `v` remap loops over `i`, builds a bounded primitive from `g.f[i, :]`, and evaluates left/right velocity cell edges.  

Add this helper:

```python
def eval_bounded_primitive_batch_rows(
    cell_avg: np.ndarray,
    dx: float,
    xmin: float,
    xmax: float,
    x: np.ndarray,
) -> np.ndarray:
    """
    Batched version of eval_bounded_primitive_from_cell_averages.

    Parameters
    ----------
    cell_avg:
        Shape (n_batch, n_cells). For this code: (Nx, Nv).
    dx:
        Cell width.
    xmin, xmax:
        Physical bounds used by the original scalar function.
    x:
        Query points with shape (n_batch, n_query).

    Returns
    -------
    Primitive values with shape (n_batch, n_query).
    """
    n_batch, n = cell_avg.shape

    masses = cell_avg * dx
    p_edges = np.concatenate(
        (
            np.zeros((n_batch, 1), dtype=cell_avg.dtype),
            np.cumsum(masses, axis=1),
        ),
        axis=1,
    )

    total = p_edges[:, -1]
    x_edges = xmin + np.arange(n + 1) * dx

    spline = CubicSpline(
        x_edges,
        p_edges.T,
        axis=0,
        bc_type="natural",
        extrapolate=False,
    )

    vals = eval_cubic_components_axis0(
        spline,
        np.clip(x.T, xmin, xmax),
    ).T

    return np.where(
        x >= xmax,
        total[:, None],
        np.where(x > xmin, vals, 0.0),
    )
```

Then rewrite `advance_v_conservative` as:

```python
def advance_v_conservative(g: Grid, h: float):
    tau = g.DT * h

    edges = g.vmin + np.arange(g.Nv + 1) * g.dv
    shift = g.force[:, None] * tau

    left = edges[:-1][None, :] - shift
    right = edges[1:][None, :] - shift

    g_left = eval_bounded_primitive_batch_rows(
        g.f,
        g.dv,
        g.vmin,
        g.vmax,
        left,
    )

    g_right = eval_bounded_primitive_batch_rows(
        g.f,
        g.dv,
        g.vmin,
        g.vmax,
        right,
    )

    g.f = (g_right - g_left) / g.dv
```

This removes the loop over `Nx`.

---

## Parts that should not be rewritten with NumPy batching

### `run` time loops

These loops should remain explicit:

```python
for top in range(1, n_top+1):
    ...
    for _ in range(n_steps-1):
        ...
```

Each iteration updates `g.f`, recomputes `rho`, `Mx`, `My`, and `force`, then uses those values in the next semi-Lagrangian step. That is a sequential time integrator, not an independent batch operation. 

### HDF5 creation/appending loops

These are not numerical hot loops:

```python
for k, v in {...}.items():
    ...
for n in obs_scalars[1:]:
    ...
for k, v in vals.items():
    ...
```

They write metadata or append separate HDF5 datasets, so NumPy batching would not materially improve them.  

### `make_video.py`

There is no meaningful NumPy-for-loop conversion needed there. Frame iteration is already handled through `FuncAnimation`; the user callback only updates one image and title per frame. 

---

## Practical recommendation

Apply the rewrites in this order:

1. `advance_x_conservative`
2. `advance_v_conservative`
3. `advance_x`
4. `advance_v`

The conservative pair matters most for the checked-in configuration. The spline pair is still worth rewriting if you frequently run with `scheme = "strang_spline"`.

[1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html "CubicSpline — SciPy v1.17.0 Manual"
