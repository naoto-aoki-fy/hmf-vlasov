# HMF Vlasov

`simulator.py` provides a Python implementation of a Strang-split conservative semi-Lagrangian HMF Vlasov solver, modeled after the workflow used by `vmf90`.

## Generate WBIC initial condition

Use `generate_wbic_init.py` to create a water-bag initial condition file for `simulator.py`.

```bash
python generate_wbic_init.py \
  --output init.npy \
  --ntheta 256 \
  --np 512 \
  --pmax 2.5 \
  --delta-p 1.3747727084867518 \
  --delta-theta 1.8954942670339805
```

The generated distribution is uniform inside:

- `p in [-delta-p/2, +delta-p/2]`
- `theta in [-delta-theta/2, +delta-theta/2]` (on the simulation domain `[-pi, pi)`)

## Run

```bash
python simulator.py \
  --init init.npy \
  --output final_state.npz \
  --ntheta 256 \
  --np 512 \
  --pmax 2.5 \
  --dt 0.1 \
  --tfinal 200.0 \
  --video evolution.mp4 \
  --video-step 5
```

- `init.npy` must contain a 2D NumPy array of shape `(ntheta, np)` with arbitrary initial data sampled as cell averages.
- Output `final_state.npz` contains:
  - `f_final`: final distribution
  - `diagnostics_initial` / `diagnostics_final`: JSON diagnostics
  - `nsteps`, `dt`, `tfinal`
- Optional video output:
  - `--video evolution.mp4`: write an MP4 of the evolving distribution.
  - Horizontal axis is `theta` and vertical axis is `p`.
  - `--video-step N`: record one frame every `N` steps (default: `1`).
  - `--video-fps FPS`: output framerate (default: `20`).
