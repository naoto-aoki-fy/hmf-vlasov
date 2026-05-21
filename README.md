# HMF Vlasov

Python implementation of the **HMF Vlasov simulation** from [`pdebuyl/vmf90`](https://github.com/pdebuyl/vmf90), using NumPy + h5py.

## Run

```bash
python hmf_vlasov.py HMF_in --output hmf.h5
```

## Notes on compatibility

This implementation reproduces the vmf90 HMF splitting loop and writes an H5MD-like structure with:

- `/h5md` metadata attributes
- `/parameters/*`
- `/observables/*/{step,time,value}` for mass/energy/etc.
- `/fields/*/{step,time,value}` for `f`, `rho`, `phi`

The included `HMF_in` comes from vmf90's `scripts/HMF_in.resonances`.
