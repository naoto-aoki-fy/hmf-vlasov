## Implementation Specification: HMF Vlasov Simulation

Use a **second-order Strang splitting** combined with a **conservative semi-Lagrangian finite-volume advection scheme**.

### 1. Model

Solve

```math
\partial_t f + p,\partial_\theta f + F(\theta,t),\partial_p f = 0,
```

with

```math
F(\theta,t)=-M_x(t)\sin\theta + M_y(t)\cos\theta,
```

```math
M_x=\iint f(\theta,p,t)\cos\theta,d\theta dp,
\qquad
M_y=\iint f(\theta,p,t)\sin\theta,d\theta dp.
```

The domain is

```math
\theta\in[0,2\pi),\qquad p\in[-p_{\max},p_{\max}].
```

Use periodic boundary conditions in (\theta). Outside the truncated momentum domain, set (f=0).

---

## 2. Discrete Variables

Use a uniform cell-centered grid:

```math
\theta_i=\left(i+\frac12\right)\Delta\theta,
\qquad
\Delta\theta=\frac{2\pi}{N_\theta},
```

```math
p_j=-p_{\max}+\left(j+\frac12\right)\Delta p,
\qquad
\Delta p=\frac{2p_{\max}}{N_p}.
```

The numerical unknown is the cell average

```math
\bar f_{i,j}^n
\approx
\frac{1}{\Delta\theta\Delta p}
\int_{\theta_{i-1/2}}^{\theta_{i+1/2}}
\int_{p_{j-1/2}}^{p_{j+1/2}}
f(\theta,p,t_n),dp,d\theta.
```

The total mass is

```math
L_1^n=
\sum_{i,j}\bar f_{i,j}^n\Delta\theta\Delta p.
```

It should remain equal to one up to round-off error.

---

## 3. Time Integration

One full time step from $t_n$ to $t_{n+1}=t_n+\Delta t$ is

```math
f^{n+1}
=
A_{\Delta t/2}
B_{\Delta t}
A_{\Delta t/2}
f^n,
```

where

```math
A:\quad \partial_t f+p,\partial_\theta f=0,
```

```math
B:\quad \partial_t f+F(\theta),\partial_p f=0.
```

The algorithm is:

1. Advect in $\theta$ for $\Delta t/2$.
2. Compute $M_x,M_y$ from the intermediate distribution.
3. Compute $F(\theta)$.
4. Advect in $p$ for $\Delta t$.
5. Advect in $\theta$ again for $\Delta t/2$.

---

## 4. Conservative Semi-Lagrangian Advection

Each split substep reduces to

```math
\partial_t g + a,\partial_x g=0.
```

For each target cell $I_k=[x_{k-1/2},x_{k+1/2}]$, trace the cell backward:

```math
I_k^\star = [x_{k-1/2}-a\tau,;x_{k+1/2}-a\tau].
```

The updated cell average is

```math
\bar g_k^{,new} = \frac{1}{\Delta x} \int_{I_k^\star} R[g^{old}](x),dx,
```

where (R[g]) is a conservative reconstruction from old cell averages.

Equivalently, using the primitive

```math
G(x)=\int^x R[g](s),ds,
```

```math
\bar g_k^{,new}
=
\frac{
G(x_{k+1/2}-a\tau)
-
G(x_{k-1/2}-a\tau)
}{\Delta x}.
```

The reconstruction must satisfy:

```math
\frac{1}{\Delta x}
\int_{I_k} R[g](x),dx
=
\bar g_k.
```

Recommended reconstruction options:

| Option                             |                 Order | Notes                           |
| ---------------------------------- | --------------------: | ------------------------------- |
| piecewise parabolic reconstruction |           third order | robust and common               |
| conservative cubic reconstruction  | third or fourth order | simple for smooth cases         |
| conservative WENO reconstruction   |            high order | better for sharp waterbag edges |

For publication-quality simulations, use at least third-order conservative reconstruction.

---

## 5. (\theta)-Advection Substep

For fixed (p_j),

```math
\partial_t f + p_j\partial_\theta f=0.
```

For a substep length (\tau),

```math
a=p_j.
```

The displacement is

```math
d_j=p_j\tau.
```

For each angular cell (i), use the conservative semi-Lagrangian formula

```math
\bar f_{i,j}^{new}
=
\frac{1}{\Delta\theta}
\int_{\theta_{i-1/2}-d_j}^{\theta_{i+1/2}-d_j}
R_\theta[\bar f_{\cdot,j}^{old}](\theta),d\theta.
```

Apply periodic wrapping:

```math
\theta \mapsto \theta \bmod 2\pi.
```

This substep must conserve the mass of each fixed-(p_j) row.

---

## 6. Force Evaluation

After the first (\theta)-half-step, compute

```math
M_x=
\sum_{i,j}
\bar f_{i,j}^{*}
\cos\theta_i
\Delta\theta\Delta p,
```

```math
M_y=
\sum_{i,j}
\bar f_{i,j}^{*}
\sin\theta_i
\Delta\theta\Delta p.
```

Then compute

```math
F_i=-M_x\sin\theta_i+M_y\cos\theta_i.
```

For higher quadrature accuracy, replace (\cos\theta_i) and (\sin\theta_i) by their cell averages:

```math
\langle \cos\theta\rangle_i
=
\frac{1}{\Delta\theta}
\int_{\theta_{i-1/2}}^{\theta_{i+1/2}}
\cos\theta,d\theta,
```

```math
\langle \sin\theta\rangle_i
=
\frac{1}{\Delta\theta}
\int_{\theta_{i-1/2}}^{\theta_{i+1/2}}
\sin\theta,d\theta.
```

---

## 7. (p)-Advection Substep

For fixed (\theta_i),

```math
\partial_t f+F_i\partial_p f=0.
```

For substep length (\tau=\Delta t),

```math
a=F_i,
\qquad
d_i=F_i\Delta t.
```

For each momentum cell (j),

```math
\bar f_{i,j}^{new}
=
\frac{1}{\Delta p}
\int_{p_{j-1/2}-d_i}^{p_{j+1/2}-d_i}
R_p[\bar f_{i,\cdot}^{old}](p),dp.
```

At the momentum boundary, use zero extension:

```math
f(\theta,p,t)=0
\quad\text{for}\quad
p\notin[-p_{\max},p_{\max}].
```

The chosen (p_{\max}) must be large enough that boundary mass remains negligible.

---

## 8. Full Step Specification

Given (\bar f^n):

### Step 1: First angular half-step

```math
\bar f^{*}
=
A_{\Delta t/2}\bar f^n.
```

### Step 2: Magnetization

```math
M_x^{*}
=
\sum_{i,j}
\bar f_{i,j}^{*}
\cos\theta_i\Delta\theta\Delta p,
```

```math
M_y^{*}
=
\sum_{i,j}
\bar f_{i,j}^{*}
\sin\theta_i\Delta\theta\Delta p.
```

### Step 3: Force

```math
F_i^{*}
=
-M_x^{*}\sin\theta_i
+
M_y^{*}\cos\theta_i.
```

### Step 4: Momentum full-step

```math
\bar f^{**}
=
B_{\Delta t}(F^*)\bar f^{*}.
```

### Step 5: Second angular half-step

```math
\bar f^{n+1}
=
A_{\Delta t/2}\bar f^{**}.
```

---

## 9. Diagnostics

At every output time, compute:

### Mass

```math
L_1(t)=\sum_{i,j}\bar f_{i,j}\Delta\theta\Delta p.
```

### Energy

```math
U(t)=
\sum_{i,j}
\frac{p_j^2}{2}
\bar f_{i,j}\Delta\theta\Delta p
+
\frac{1-M_x^2-M_y^2}{2}.
```

### Magnetization

```math
M(t)=\sqrt{M_x^2+M_y^2}.
```

### (L_2) norm

```math
L_2(t)=
\sum_{i,j}
\bar f_{i,j}^2\Delta\theta\Delta p.
```

### Positivity check

```math
f_{\min}(t)=\min_{i,j}\bar f_{i,j}.
```

### Momentum-boundary mass

```math
B_p(t)
=
\sum_{i,;j\in\mathcal B}
\bar f_{i,j}\Delta\theta\Delta p,
```

where (\mathcal B) denotes the outermost few momentum cells.

---

## 10. Accuracy Requirements

For a publication-quality run, report:

```math
\max_t
\left|
\frac{L_1(t)-L_1(0)}{L_1(0)}
\right|,
```

```math
\max_t
\left|
\frac{U(t)-U(0)}{U(0)}
\right|,
```

```math
\max_t |f_{\min}(t)|,
```

```math
\max_t B_p(t).
```

Also perform convergence checks using at least two refinements:

```math
(N_\theta,N_p,\Delta t),
```

```math
(2N_\theta,2N_p,\Delta t/2),
```

```math
(4N_\theta,4N_p,\Delta t/4).
```

Compare (M(t)), (U(t)), and (L_2(t)).

---

## 14. Minimal Algorithm Statement

The complete method can be stated as:

> The HMF Vlasov equation is solved on a uniform finite-volume grid in ((\theta,p)). Time integration is performed by second-order Strang splitting. Each split transport equation is solved by a conservative semi-Lagrangian advection operator based on a polynomial reconstruction of cell averages. The angular advection uses periodic wrapping, while the momentum advection uses zero extension outside the truncated momentum interval. The self-consistent force is evaluated from the magnetization computed after the first angular half-step. Mass, energy, magnetization, (L_2), positivity, and momentum-boundary mass are monitored to assess numerical accuracy.
