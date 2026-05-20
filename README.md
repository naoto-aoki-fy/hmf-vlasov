# HMF Vlasov

`simulator.py` provides a Python implementation of a Strang-split conservative semi-Lagrangian HMF Vlasov solver, modeled after the workflow used by `vmf90`.

## Run

```bash
python simulator.py \
  --init init.npy \
  --output final_state.npz \
  --ntheta 256 \
  --np 256 \
  --pmax 8.0 \
  --dt 0.05 \
  --tfinal 10.0 \
  --video evolution.mp4
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
