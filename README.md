# HMF Vlasov

C++ finite-volume semi-Lagrangian simulator for the HMF Vlasov equation.

## Build

```bash
make
```

## Run

```bash
./simulator input.bin output.bin dt nsteps
```

## Binary state format

The simulator reads and writes a binary dump with this layout:

1. Header (`sizeof(Header)` bytes, little-endian native layout):
   - `char magic[8]` (must start with `"HMFV1"`)
   - `uint32_t ntheta`
   - `uint32_t np`
   - `double pmax`
   - `double time`
2. `ntheta*np` doubles with cell averages `f[i*np + j]`.

Grid conventions:
- `theta_i = (i+1/2) * 2*pi/ntheta`
- `p_j = -pmax + (j+1/2) * 2*pmax/np`
