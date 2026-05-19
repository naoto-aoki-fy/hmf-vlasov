#!/usr/bin/env python3
"""HMF Vlasov simulator using Strang splitting and conservative semi-Lagrangian advection.

Binary dump format (little-endian):
- magic: 8 bytes = b'HMFVLSV1'
- N_theta: uint32
- N_p: uint32
- p_max: float64
- t: float64
- payload: N_theta * N_p float64 cell averages in C order (theta major)
"""
from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MAGIC = b"HMFVLSV1"
HEADER_STRUCT = struct.Struct("<8sIIdd")


@dataclass
class Grid:
    n_theta: int
    n_p: int
    p_max: float

    @property
    def dtheta(self) -> float:
        return 2.0 * np.pi / self.n_theta

    @property
    def dp(self) -> float:
        return 2.0 * self.p_max / self.n_p

    @property
    def theta_centers(self) -> np.ndarray:
        return (np.arange(self.n_theta, dtype=np.float64) + 0.5) * self.dtheta

    @property
    def p_centers(self) -> np.ndarray:
        return -self.p_max + (np.arange(self.n_p, dtype=np.float64) + 0.5) * self.dp


def dump_state(path: Path, grid: Grid, t: float, f: np.ndarray) -> None:
    if f.shape != (grid.n_theta, grid.n_p):
        raise ValueError("State shape does not match grid")
    with path.open("wb") as fh:
        fh.write(HEADER_STRUCT.pack(MAGIC, grid.n_theta, grid.n_p, grid.p_max, t))
        fh.write(np.asarray(f, dtype=np.float64, order="C").tobytes(order="C"))


def load_state(path: Path) -> Tuple[Grid, float, np.ndarray]:
    data = path.read_bytes()
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("Invalid state file: too short")
    magic, n_theta, n_p, p_max, t = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("Invalid state file: bad magic")
    expected = HEADER_STRUCT.size + n_theta * n_p * 8
    if len(data) != expected:
        raise ValueError(f"Invalid state file length: expected {expected}, got {len(data)}")
    arr = np.frombuffer(data, dtype="<f8", count=n_theta * n_p, offset=HEADER_STRUCT.size)
    f = np.array(arr.reshape((n_theta, n_p), order="C"), copy=True)
    return Grid(n_theta=n_theta, n_p=n_p, p_max=p_max), t, f


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    same_sign = (a * b) > 0.0
    out[same_sign] = np.sign(a[same_sign]) * np.minimum(np.abs(a[same_sign]), np.abs(b[same_sign]))
    return out


def semi_lagrangian_1d(values: np.ndarray, dx: float, displacement: float, periodic: bool) -> np.ndarray:
    """Second-order conservative semi-Lagrangian advection for cell averages."""
    n = values.shape[0]
    slope = minmod(values - np.roll(values, 1), np.roll(values, -1) - values)

    def primitive(x: np.ndarray) -> np.ndarray:
        xi = x / dx
        kf = np.floor(xi).astype(np.int64)
        frac = xi - kf

        if periodic:
            k = np.mod(kf, n)
            turns = np.floor_divide(kf, n)
            base = turns * np.sum(values) * dx
            prefix = np.concatenate(([0.0], np.cumsum(values * dx)))
            y = values[k] + 0.5 * slope[k] * (2.0 * frac - 1.0)
            return base + prefix[k] + frac * y * dx

        prefix = np.concatenate(([0.0], np.cumsum(values * dx)))
        inside = (kf >= 0) & (kf < n)
        k = np.clip(kf, 0, n - 1)
        y = np.where(inside, values[k] + 0.5 * slope[k] * (2.0 * frac - 1.0), 0.0)
        x_clamped = np.clip(x, 0.0, n * dx)
        kx = np.floor(x_clamped / dx).astype(np.int64)
        kx = np.clip(kx, 0, n - 1)
        fracx = (x_clamped / dx) - kx
        return prefix[kx] + fracx * (values[kx] + 0.5 * slope[kx] * (2.0 * fracx - 1.0)) * dx

    edges = np.arange(n + 1, dtype=np.float64) * dx - displacement
    g = primitive(edges)
    return (g[1:] - g[:-1]) / dx


def advect_theta(f: np.ndarray, grid: Grid, tau: float) -> np.ndarray:
    out = np.empty_like(f)
    p = grid.p_centers
    for j in range(grid.n_p):
        out[:, j] = semi_lagrangian_1d(f[:, j], grid.dtheta, p[j] * tau, periodic=True)
    return out


def advect_p(f: np.ndarray, grid: Grid, tau: float, force: np.ndarray) -> np.ndarray:
    out = np.empty_like(f)
    for i in range(grid.n_theta):
        out[i, :] = semi_lagrangian_1d(f[i, :], grid.dp, force[i] * tau, periodic=False)
    return out


def magnetization(f: np.ndarray, grid: Grid) -> Tuple[float, float]:
    theta = grid.theta_centers
    weight = grid.dtheta * grid.dp
    mx = np.sum(f * np.cos(theta)[:, None]) * weight
    my = np.sum(f * np.sin(theta)[:, None]) * weight
    return float(mx), float(my)


def diagnostics(f: np.ndarray, grid: Grid, boundary_cells: int = 2) -> Dict[str, float]:
    theta = grid.theta_centers
    p = grid.p_centers
    weight = grid.dtheta * grid.dp
    mass = float(np.sum(f) * weight)
    mx, my = magnetization(f, grid)
    kinetic = float(np.sum(0.5 * (p[None, :] ** 2) * f) * weight)
    potential = 0.5 * (1.0 - mx * mx - my * my)
    l2 = float(np.sum(f * f) * weight)
    b = min(boundary_cells, grid.n_p // 2)
    boundary_mass = float(np.sum(f[:, :b] + f[:, -b:]) * weight)
    return {
        "mass": mass,
        "energy": kinetic + potential,
        "Mx": mx,
        "My": my,
        "M": float(np.hypot(mx, my)),
        "L2": l2,
        "fmin": float(np.min(f)),
        "boundary_mass": boundary_mass,
    }


def step_strang(f: np.ndarray, grid: Grid, dt: float) -> np.ndarray:
    half = advect_theta(f, grid, dt * 0.5)
    mx, my = magnetization(half, grid)
    theta = grid.theta_centers
    force = -mx * np.sin(theta) + my * np.cos(theta)
    full = advect_p(half, grid, dt, force)
    return advect_theta(full, grid, dt * 0.5)


def run_simulation(grid: Grid, f0: np.ndarray, t0: float, dt: float, steps: int, diag_every: int) -> Tuple[float, np.ndarray]:
    t = t0
    f = np.array(f0, copy=True)
    for n in range(1, steps + 1):
        f = step_strang(f, grid, dt)
        t += dt
        if diag_every > 0 and (n % diag_every == 0 or n == steps):
            d = diagnostics(f, grid)
            print(
                f"step={n:6d} t={t:.6f} mass={d['mass']:.12e} energy={d['energy']:.12e} "
                f"M={d['M']:.12e} L2={d['L2']:.12e} fmin={d['fmin']:.12e} bmass={d['boundary_mass']:.12e}"
            )
    return t, f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMF Vlasov simulator")
    p.add_argument("--input", type=Path, required=True, help="Input binary state dump")
    p.add_argument("--output", type=Path, required=True, help="Output binary state dump")
    p.add_argument("--dt", type=float, required=True, help="Time step")
    p.add_argument("--steps", type=int, required=True, help="Number of time steps")
    p.add_argument("--diag-every", type=int, default=10, help="Print diagnostics every N steps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid, t0, f0 = load_state(args.input)
    tf, ff = run_simulation(grid, f0, t0, args.dt, args.steps, args.diag_every)
    dump_state(args.output, grid, tf, ff)


if __name__ == "__main__":
    main()
