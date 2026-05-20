#!/usr/bin/env python3
"""HMF Vlasov simulator inspired by vmf90 splitting workflow.

Input: arbitrary initial phase-space data on a cell-centered (theta, p) grid.
Output: final state after time evolution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Grid:
    ntheta: int
    np_: int
    pmax: float

    @property
    def dtheta(self) -> float:
        return 2.0 * np.pi / self.ntheta

    @property
    def dp(self) -> float:
        return 2.0 * self.pmax / self.np_

    @property
    def theta_centers(self) -> np.ndarray:
        return (np.arange(self.ntheta) + 0.5) * self.dtheta

    @property
    def p_centers(self) -> np.ndarray:
        return -self.pmax + (np.arange(self.np_) + 0.5) * self.dp


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    mask = a * b > 0.0
    out[mask] = np.sign(a[mask]) * np.minimum(np.abs(a[mask]), np.abs(b[mask]))
    return out


def primitive_from_piecewise_linear(values: np.ndarray, dx: float, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    n = values.size
    if periodic:
        left = np.roll(values, 1)
        right = np.roll(values, -1)
        slopes = minmod(values - left, right - values)
    else:
        slopes = np.zeros_like(values)
        slopes[1:-1] = minmod(values[1:-1] - values[:-2], values[2:] - values[1:-1])

    interfaces = np.arange(n + 1, dtype=float) * dx
    cell_int = values * dx
    pref = np.concatenate(([0.0], np.cumsum(cell_int)))

    def prim(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        if periodic:
            period = n * dx
            xx = np.mod(xx, period)
        out = np.zeros_like(xx)
        mask_low = xx <= 0.0
        mask_high = xx >= n * dx
        mask_mid = ~(mask_low | mask_high)
        out[mask_high] = pref[-1]
        if np.any(mask_mid):
            xm = xx[mask_mid]
            k = np.floor(xm / dx).astype(int)
            k = np.clip(k, 0, n - 1)
            x0 = interfaces[k]
            xi = xm - (x0 + 0.5 * dx)
            out[mask_mid] = pref[k] + values[k] * (xm - x0) + 0.5 * slopes[k] / dx * (xi**2 - (0.5 * dx) ** 2)
        return out

    return interfaces, prim


def advect_1d_conservative(values: np.ndarray, shift: float, dx: float, periodic: bool) -> np.ndarray:
    return advect_1d_conservative_batch(values, np.array(shift), dx=dx, periodic=periodic)


def advect_1d_conservative_batch(
    values: np.ndarray, shifts: np.ndarray, dx: float, periodic: bool, axis: int = 0
) -> np.ndarray:
    moved = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    n = moved.shape[0]
    batch_shape = moved.shape[1:]

    shifts_arr = np.asarray(shifts, dtype=float)
    if shifts_arr.shape != batch_shape:
        raise ValueError(f"shifts shape {shifts_arr.shape} must match batch shape {batch_shape}")

    if periodic:
        left = np.roll(moved, 1, axis=0)
        right = np.roll(moved, -1, axis=0)
        slopes = minmod(moved - left, right - moved)
    else:
        slopes = np.zeros_like(moved)
        slopes[1:-1, ...] = minmod(moved[1:-1, ...] - moved[:-2, ...], moved[2:, ...] - moved[1:-1, ...])

    pref = np.concatenate((np.zeros((1,) + batch_shape), np.cumsum(moved * dx, axis=0)), axis=0)

    interfaces = np.arange(n + 1, dtype=float) * dx
    left_x = interfaces[:-1].reshape((n,) + (1,) * len(batch_shape)) - shifts_arr
    right_x = interfaces[1:].reshape((n,) + (1,) * len(batch_shape)) - shifts_arr

    def prim_eval(x: np.ndarray) -> np.ndarray:
        xx = x
        if periodic:
            xx = np.mod(xx, n * dx)

        out = np.zeros_like(xx)
        mask_low = xx <= 0.0
        mask_high = xx >= n * dx
        mask_mid = ~(mask_low | mask_high)
        if np.any(mask_high):
            out[mask_high] = np.broadcast_to(pref[-1], xx.shape)[mask_high]

        if np.any(mask_mid):
            k = np.floor(xx / dx).astype(int)
            k = np.clip(k, 0, n - 1)
            x0 = k * dx
            xi = xx - (x0 + 0.5 * dx)
            pref_k = np.take_along_axis(pref[:-1], k, axis=0)
            val_k = np.take_along_axis(moved, k, axis=0)
            slope_k = np.take_along_axis(slopes, k, axis=0)
            mid_val = pref_k + val_k * (xx - x0) + 0.5 * slope_k / dx * (xi**2 - (0.5 * dx) ** 2)
            out = np.where(mask_mid, mid_val, out)
        return out

    advected = (prim_eval(right_x) - prim_eval(left_x)) / dx
    return np.moveaxis(advected, 0, axis)


def theta_step(f: np.ndarray, grid: Grid, tau: float) -> np.ndarray:
    return advect_1d_conservative_batch(f, shifts=grid.p_centers * tau, dx=grid.dtheta, periodic=True, axis=0)


def magnetization(f: np.ndarray, grid: Grid) -> tuple[float, float]:
    th = grid.theta_centers[:, None]
    weight = grid.dtheta * grid.dp
    mx = float(np.sum(f * np.cos(th)) * weight)
    my = float(np.sum(f * np.sin(th)) * weight)
    return mx, my


def p_step(f: np.ndarray, force_theta: np.ndarray, grid: Grid, tau: float) -> np.ndarray:
    return advect_1d_conservative_batch(f, shifts=force_theta * tau, dx=grid.dp, periodic=False, axis=1)


def strang_step(f: np.ndarray, grid: Grid, dt: float) -> np.ndarray:
    f1 = theta_step(f, grid, 0.5 * dt)
    mx, my = magnetization(f1, grid)
    th = grid.theta_centers
    force = -mx * np.sin(th) + my * np.cos(th)
    f2 = p_step(f1, force, grid, dt)
    return theta_step(f2, grid, 0.5 * dt)


def diagnostics(f: np.ndarray, grid: Grid) -> dict[str, float]:
    mx, my = magnetization(f, grid)
    weight = grid.dtheta * grid.dp
    kinetic = float(np.sum((0.5 * grid.p_centers[None, :] ** 2) * f) * weight)
    energy = kinetic + 0.5 * (1.0 - mx**2 - my**2)
    return {
        "mass": float(np.sum(f) * weight),
        "mx": mx,
        "my": my,
        "M": float(np.sqrt(mx**2 + my**2)),
        "energy": energy,
        "l2": float(np.sum(f**2) * weight),
        "f_min": float(np.min(f)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Time-evolve HMF Vlasov initial data and dump final state.")
    p.add_argument("--init", type=Path, required=True, help="Input .npy file containing f(theta, p)")
    p.add_argument("--output", type=Path, required=True, help="Output .npz final dump")
    p.add_argument("--ntheta", type=int, required=True)
    p.add_argument("--np", dest="np_", type=int, required=True)
    p.add_argument("--pmax", type=float, required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--tfinal", type=float, required=True)
    args = p.parse_args()

    grid = Grid(args.ntheta, args.np_, args.pmax)
    f = np.load(args.init)
    if f.shape != (grid.ntheta, grid.np_):
        raise ValueError(f"Initial data shape {f.shape} != ({grid.ntheta}, {grid.np_})")

    nsteps = int(np.round(args.tfinal / args.dt))
    if not np.isclose(nsteps * args.dt, args.tfinal):
        raise ValueError("tfinal must be an integer multiple of dt")

    diag0 = diagnostics(f, grid)
    for _ in range(nsteps):
        f = strang_step(f, grid, args.dt)
    diagf = diagnostics(f, grid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        f_final=f,
        diagnostics_initial=json.dumps(diag0),
        diagnostics_final=json.dumps(diagf),
        nsteps=nsteps,
        dt=args.dt,
        tfinal=args.tfinal,
    )


if __name__ == "__main__":
    main()
