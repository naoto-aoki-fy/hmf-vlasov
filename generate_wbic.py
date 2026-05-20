#!/usr/bin/env python3
"""Generate a WBIC (waterbag) initial-condition state file."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simulator import Grid, dump_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate WBIC initial condition binary state")
    p.add_argument("--output", type=Path, required=True, help="Output binary state path")
    p.add_argument("--n-theta", type=int, required=True, help="Number of theta cells")
    p.add_argument("--n-p", type=int, required=True, help="Number of p cells")
    p.add_argument("--p-max", type=float, required=True, help="Maximum |p| of simulation domain")
    p.add_argument("--delta-p", type=float, required=True, help="WBIC width in p (from -delta_p/2 to +delta_p/2)")
    p.add_argument("--delta-theta", type=float, required=True, help="WBIC width in theta (from -delta_theta/2 to +delta_theta/2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_theta <= 0 or args.n_p <= 0:
        raise ValueError("--n-theta and --n-p must be positive")
    if args.p_max <= 0.0:
        raise ValueError("--p-max must be positive")
    if args.delta_p <= 0.0:
        raise ValueError("--delta-p must be positive")
    if args.delta_theta <= 0.0:
        raise ValueError("--delta-theta must be positive")
    if args.delta_p > 2.0 * args.p_max:
        raise ValueError("--delta-p cannot exceed total p-domain width 2*p_max")
    if args.delta_theta > 2.0 * np.pi:
        raise ValueError("--delta-theta cannot exceed 2*pi")

    grid = Grid(n_theta=args.n_theta, n_p=args.n_p, p_max=args.p_max)
    theta = grid.theta_centers[:, None]
    p = grid.p_centers[None, :]

    in_p = np.abs(p) <= (0.5 * args.delta_p)
    theta_wrapped = (theta + np.pi) % (2.0 * np.pi) - np.pi
    in_theta = np.abs(theta_wrapped) <= (0.5 * args.delta_theta)

    f = (in_theta & in_p).astype(np.float64)
    mass = float(np.sum(f) * grid.dtheta * grid.dp)
    if mass <= 0.0:
        raise ValueError("No grid cell fell inside the WBIC support; increase resolution or widen deltas")
    f /= mass

    dump_state(args.output, grid, t=0.0, f=f)


if __name__ == "__main__":
    main()
