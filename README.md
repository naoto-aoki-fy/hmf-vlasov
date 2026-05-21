# HMF Vlasov

Python implementation of the **HMF Vlasov simulation** from [`pdebuyl/vmf90`](https://github.com/pdebuyl/vmf90), using NumPy + h5py.

## Run

```bash
python hmf_vlasov.py HMF_in --output hmf.h5
```


## Configuration

`HMF_in` is now parsed as **TOML**. Use quoted strings and standard TOML numbers/booleans.

## Notes on compatibility

This implementation reproduces the vmf90 HMF splitting loop and writes an H5MD-like structure with:

- `/h5md` metadata attributes
- `/parameters/*`
- `/observables/*/{step,time,value}` for mass/energy/etc.
- `/fields/*/{step,time,value}` for `f`, `rho`, `phi`

The included `HMF_in` comes from vmf90's `scripts/HMF_in.resonances`.

## Create an animation

Create a video (MP4) of the time evolution of the phase-space distribution:

```bash
python make_video.py hmf.h5 --output distribution.mp4
```

Or create a GIF:

```bash
python make_video.py hmf.h5 --output distribution.gif
```
