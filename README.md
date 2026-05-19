# HMF Vlasov

## C++ simulator

Build:

```bash
g++ -O2 -std=c++17 simulator.cpp -o simulator
```

Run:

```bash
./simulator NTHETA NP PMAX DT NSTEPS input.txt output.txt
```

`input.txt` must contain `NTHETA * NP` floating-point values in row-major order, one theta-row per line (values separated by spaces). The program writes the final distribution to `output.txt` in the same layout.
