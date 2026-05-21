#!/usr/bin/env python3
"""Generate water-bag initial condition (WBIC) on the simulator grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def theta_centers(ntheta: int) -> np.ndarray:
    dtheta = 2.0 * np.pi / ntheta
    return -np.pi + (np.arange(ntheta) + 0.5) * dtheta


def p_centers(np_: int, pmax: float) -> np.ndarray:
    dp = 2.0 * pmax / np_
    return -pmax + (np.arange(np_) + 0.5) * dp


def generate_wbic(ntheta: int, np_: int, pmax: float, delta_theta: float, delta_p: float) -> np.ndarray:
    if ntheta <= 0 or np_ <= 0:
        raise ValueError("ntheta and np must be positive")
    if pmax <= 0:
        raise ValueError("pmax must be positive")
    if not (0.0 < delta_theta <= 2.0 * np.pi):
        raise ValueError("delta-theta must satisfy 0 < delta-theta <= 2*pi")
    if not (0.0 < delta_p <= 2.0 * pmax):
        raise ValueError("delta-p must satisfy 0 < delta-p <= 2*pmax")

    th = theta_centers(ntheta)
    pp = p_centers(np_, pmax)

    theta_mask = np.abs(th) <= (0.5 * delta_theta)
    p_mask = np.abs(pp) <= (0.5 * delta_p)

    f = np.zeros((ntheta, np_), dtype=float)
    f[np.ix_(theta_mask, p_mask)] = 1.0 / (delta_theta * delta_p)
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WBIC .npy initial condition for simulator.py")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy file")
    parser.add_argument("--ntheta", type=int, required=True)
    parser.add_argument("--np", dest="np_", type=int, required=True)
    parser.add_argument("--pmax", type=float, required=True)
    parser.add_argument("--delta-theta", type=float, required=True)
    parser.add_argument("--delta-p", type=float, required=True)
    args = parser.parse_args()

    f = generate_wbic(args.ntheta, args.np_, args.pmax, args.delta_theta, args.delta_p)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, f)


if __name__ == "__main__":
    main()
