# hmf-vlasov

Python simulator for the HMF Vlasov equation, based on the method in `specification.md`:

- Second-order Strang splitting (`A(dt/2) -> B(dt) -> A(dt/2)`).
- Conservative semi-Lagrangian finite-volume advection in both `theta` and `p` directions.
- Periodic angular boundary and zero-extension momentum boundary.
- Runtime diagnostics (mass, energy, magnetization, L2, minimum value, boundary mass).

## Requirements

- Python 3.10+
- NumPy

## Binary state format

State dumps are little-endian binary files with this layout:

1. `magic` (8 bytes): `HMFVLSV1`
2. `N_theta` (`uint32`)
3. `N_p` (`uint32`)
4. `p_max` (`float64`)
5. `t` (`float64`)
6. `f` payload (`N_theta * N_p` values of `float64` in C order, shape `(N_theta, N_p)`)

## Run simulator

```bash
python simulator.py \
  --input initial.bin \
  --output final.bin \
  --dt 0.1 \
  --steps 1000 \
  --diag-every 5 \
  --video distribution.mp4 \
  --video-every 5 \
  --video-fps 30
```

## Quick utility snippet to create an initial dump

```python
import numpy as np
from pathlib import Path
from simulator import Grid, dump_state

grid = Grid(n_theta=64, n_p=64, p_max=4.0)
theta = grid.theta_centers[:, None]
p = grid.p_centers[None, :]
f0 = np.exp(-p**2 / 2.0) * (1.0 + 0.1 * np.cos(theta))
# normalize mass to 1
f0 /= (f0.sum() * grid.dtheta * grid.dp)

dump_state(Path("initial.bin"), grid, t=0.0, f=f0)
```



If `--video` is set, the simulator records the evolving distribution and writes an animation. The horizontal axis is `theta` (range `[-pi, pi]`) and the vertical axis is `p`. Use `.mp4` (ffmpeg writer) or `.gif` (pillow writer) as the output extension.


## Generate WBIC initial condition

Use `generate_wbic.py` to create a waterbag initial condition (uniform inside a rectangle in `(theta, p)`) and write it in the simulator binary format:

```bash
python generate_wbic.py \
  --output initial.bin \
  --n-theta 256 \
  --n-p 512 \
  --p-max 2.5 \
  --delta-p 1.3747727084867518 \
  --delta-theta 1.8954942670339805
```

WBIC support ranges:
- `p in [-delta_p/2, +delta_p/2]`
- `theta in [-delta_theta/2, +delta_theta/2]` (with periodic wrap to `[-pi, pi)`)

The generated distribution is normalized to unit total mass.
