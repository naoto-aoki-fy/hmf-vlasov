import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import tomllib
import numpy as np
import h5py
from scipy.interpolate import CubicSpline

PI = np.pi


def parse_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass
class Grid:
    Nx: int; Nv: int; xmin: float; xmax: float; vmin: float; vmax: float; DT: float

    def __post_init__(self):
        self.dx = (self.xmax - self.xmin) / self.Nx
        self.dv = (self.vmax - self.vmin) / self.Nv
        self.x_edges = self.xmin + np.arange(self.Nx + 1) * self.dx
        self.v_edges = self.vmin + np.arange(self.Nv + 1) * self.dv
        # vmf90 uses nodal coordinates x_i = xmin + (i-1)dx and v_m = vmin + (m-1)dv.
        # Keep x_edges/v_edges for conservative remap utilities, but align the active
        # advection grid with vmf90 so pointwise comparisons match at x=+dx/2, etc.
        self.x = self.xmin + np.arange(self.Nx) * self.dx
        self.v = self.vmin + np.arange(self.Nv) * self.dv
        self.f = np.zeros((self.Nx, self.Nv))
        self.rho = np.zeros(self.Nx)
        self.phi = np.zeros(self.Nv)
        self.force = np.zeros(self.Nx)


def init_waterbag(g: Grid, width: float, bag: float):
    mask_x = np.abs(g.x) <= width
    mask_v = np.abs(g.v) <= bag
    g.f.fill(0.0)
    g.f[np.ix_(mask_x, mask_v)] = 1.0
    norm = g.f.sum() * g.dx * g.dv
    if norm > 0.0:
        g.f /= norm


def compute_rho(g: Grid):
    g.rho = g.f.sum(axis=1) * g.dv


def compute_phi(g: Grid):
    g.phi = g.f.sum(axis=0) * g.dx


def compute_M(g: Grid):
    mx = np.sum(np.cos(g.x) * g.rho) * g.dx
    my = np.sum(np.sin(g.x) * g.rho) * g.dx
    return mx, my


def compute_force(g: Grid, mx: float, my: float):
    g.force = np.cos(g.x) * my - np.sin(g.x) * mx


def eval_cubic_components_axis0(cs: CubicSpline, xq: np.ndarray, fill_value=None) -> np.ndarray:
    xp = cs.x
    nseg = xp.size - 1

    if fill_value is None:
        valid = np.ones_like(xq, dtype=bool)
        xwork = xq
    else:
        valid = (xq >= xp[0]) & (xq <= xp[-1])
        xwork = np.clip(xq, xp[0], xp[-1])

    seg = np.searchsorted(xp, xwork, side="right") - 1
    seg = np.clip(seg, 0, nseg - 1)
    t = xwork - xp[seg]
    batch = np.arange(xq.shape[1])[None, :]

    c = cs.c
    out = ((c[0, seg, batch] * t + c[1, seg, batch]) * t + c[2, seg, batch]) * t + c[3, seg, batch]

    if fill_value is not None:
        out = np.where(valid, out, fill_value)
    return out


def advance_x(g: Grid, h: float):
    x_back = g.x[:, None] - g.DT * h * g.v[None, :]
    period = g.xmax - g.xmin
    x_eval = ((x_back - g.xmin) % period) + g.xmin

    x_periodic = np.concatenate((g.x, [g.xmax]))
    y_periodic = np.vstack((g.f, g.f[:1, :]))
    spline = CubicSpline(x_periodic, y_periodic, axis=0, bc_type="periodic", extrapolate="periodic")
    g.f = eval_cubic_components_axis0(spline, x_eval)


def advance_v(g: Grid, h: float):
    v_back = g.v[None, :] - g.DT * h * g.force[:, None]
    spline = CubicSpline(g.v, g.f.T, axis=0, bc_type="natural", extrapolate=False)
    g.f = eval_cubic_components_axis0(spline, v_back.T, fill_value=0.0).T


def eval_periodic_primitive_from_cell_averages(cell_avg: np.ndarray, dx: float, x: np.ndarray) -> np.ndarray:
    n = cell_avg.size
    l = n * dx
    masses = cell_avg * dx
    p_edges = np.zeros(n + 1)
    p_edges[1:] = np.cumsum(masses)
    total = p_edges[-1]

    x_edges = np.arange(n + 1) * dx
    q_edges = p_edges - total * (x_edges / l)
    q_spline = CubicSpline(x_edges, q_edges, bc_type="periodic", extrapolate="periodic")

    wraps = np.floor(x / l)
    x_mod = x - wraps * l
    return wraps * total + (q_spline(x_mod) + total * (x_mod / l))


def eval_periodic_primitive_batch_axis0(cell_avg: np.ndarray, dx: float, x: np.ndarray) -> np.ndarray:
    n, n_batch = cell_avg.shape
    length = n * dx

    masses = cell_avg * dx
    p_edges = np.vstack((np.zeros((1, n_batch), dtype=cell_avg.dtype), np.cumsum(masses, axis=0)))

    total = p_edges[-1, :]
    x_edges = np.arange(n + 1) * dx
    q_edges = p_edges - (x_edges[:, None] / length) * total[None, :]
    q_spline = CubicSpline(x_edges, q_edges, axis=0, bc_type="periodic", extrapolate="periodic")

    wraps = np.floor(x / length)
    x_mod = x - wraps * length
    q_val = eval_cubic_components_axis0(q_spline, x_mod)
    return wraps * total[None, :] + q_val + (x_mod / length) * total[None, :]


def eval_bounded_primitive_from_cell_averages(
    cell_avg: np.ndarray, dx: float, xmin: float, xmax: float, x: np.ndarray
) -> np.ndarray:
    n = cell_avg.size
    masses = cell_avg * dx
    p_edges = np.zeros(n + 1)
    p_edges[1:] = np.cumsum(masses)
    total = p_edges[-1]
    x_edges = xmin + np.arange(n + 1) * dx

    spline = CubicSpline(x_edges, p_edges, bc_type="natural", extrapolate=False)
    y = np.zeros_like(x, dtype=float)
    mask_mid = (x > xmin) & (x < xmax)
    y[x >= xmax] = total
    if np.any(mask_mid):
        y[mask_mid] = spline(x[mask_mid])
    return y


def eval_bounded_primitive_batch_rows(
    cell_avg: np.ndarray, dx: float, xmin: float, xmax: float, x: np.ndarray
) -> np.ndarray:
    n_batch, n = cell_avg.shape
    masses = cell_avg * dx
    p_edges = np.concatenate(
        (np.zeros((n_batch, 1), dtype=cell_avg.dtype), np.cumsum(masses, axis=1)),
        axis=1,
    )

    total = p_edges[:, -1]
    x_edges = xmin + np.arange(n + 1) * dx
    spline = CubicSpline(x_edges, p_edges.T, axis=0, bc_type="natural", extrapolate=False)
    vals = eval_cubic_components_axis0(spline, np.clip(x.T, xmin, xmax)).T
    return np.where(x >= xmax, total[:, None], np.where(x > xmin, vals, 0.0))


def advance_x_conservative(g: Grid, h: float):
    tau = g.DT * h
    edges_rel = np.arange(g.Nx + 1) * g.dx
    shift = g.v[None, :] * tau
    left = edges_rel[:-1, None] - shift
    right = edges_rel[1:, None] - shift
    g_left = eval_periodic_primitive_batch_axis0(g.f, g.dx, left)
    g_right = eval_periodic_primitive_batch_axis0(g.f, g.dx, right)
    g.f = (g_right - g_left) / g.dx


def advance_v_conservative(g: Grid, h: float):
    tau = g.DT * h
    edges = g.vmin + np.arange(g.Nv + 1) * g.dv
    shift = g.force[:, None] * tau
    left = edges[:-1][None, :] - shift
    right = edges[1:][None, :] - shift
    g_left = eval_bounded_primitive_batch_rows(g.f, g.dv, g.vmin, g.vmax, left)
    g_right = eval_bounded_primitive_batch_rows(g.f, g.dv, g.vmin, g.vmax, right)
    g.f = (g_right - g_left) / g.dv


def create_obs_group(f, root, name, data_shape, link_from=None):
    g = f.require_group(f"{root}/{name}")
    maxshape = (None, *data_shape)
    g.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int64)
    if link_from is None:
        g.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64)
    else:
        g["time"] = h5py.SoftLink(f"/{root}/{link_from}/time")
    g.create_dataset("value", shape=(0, *data_shape), maxshape=maxshape, dtype=np.float64)


def append_obs(f, root, name, step, time, value):
    g = f[f"{root}/{name}"]
    n = g["step"].shape[0]
    g["step"].resize((n+1,)); g["step"][n] = step
    if isinstance(g.get("time", getlink=True), h5py.HardLink):
        g["time"].resize((n+1,)); g["time"][n] = time
    g["value"].resize((n+1, *g["value"].shape[1:])); g["value"][n] = value


def to_vmf90_field_layout(f_cell: np.ndarray) -> np.ndarray:
    """Convert internal (Nx, Nv) cell-centered field to vmf90-style (Nv, Nx+1)."""
    return np.concatenate((f_cell.T, f_cell.T[:, :1]), axis=1)


def run(conf, out_file="hmf.h5"):
    nx, nv = int(conf["Nx"]), int(conf["Nv"])
    vmax = float(conf["vmax"])
    dt = float(conf["DT"])
    g = Grid(nx, nv, -PI, PI, -vmax, vmax, dt)
    init_waterbag(g, float(conf["width"]), float(conf["bag"]))

    n_steps, n_top = int(conf["n_steps"]), int(conf["n_top"])
    n_images = int(conf.get("n_images", n_top))
    log_every = max(1, int(conf.get("log_every", 1)))
    scheme = str(conf.get("scheme", "strang_spline")).lower()

    if scheme == "strang_spline":
        adv_x = advance_x
        adv_v = advance_v
    elif scheme == "strang_conservative":
        adv_x = advance_x_conservative
        adv_v = advance_v_conservative
    else:
        raise ValueError(f"Unknown scheme={scheme!r}. Use 'strang_spline' or 'strang_conservative'.")

    with h5py.File(out_file, "w") as f:
        h5md = f.create_group("h5md")
        h5md.attrs["author"] = "Pierre de Buyl <pdebuyl@ulb.ac.be>"
        h5md.attrs["creator"] = "vmf90_hmf"
        h5md.attrs["creator_version"] = "python-port"
        h5md.attrs["version"] = np.array([0, 1], dtype=np.int32)
        h5md.attrs["creation_time"] = int(datetime.now(timezone.utc).timestamp())
        f.create_group("trajectory")
        f.create_group("observables")
        f.create_group("parameters")

        for k, v in {
            "model": "HMF",
            "Nx": nx,
            "Nv": nv,
            "DT": dt,
            "xmin": g.xmin,
            "xmax": g.xmax,
            "vmin": g.vmin,
            "vmax": g.vmax,
            "dx": g.dx,
            "dv": g.dv,
            "scheme": scheme,
        }.items():
            f[f"parameters/{k}"] = v

        obs_scalars = ["mass","energy","en_int","en_kin","momentum","Mx","My","I2","I3","f_min","Bp"]
        create_obs_group(f, "observables", "mass", ())
        for n in obs_scalars[1:]: create_obs_group(f, "observables", n, (), link_from="mass")
        create_obs_group(f, "fields", "f", (g.Nv, g.Nx + 1))
        create_obs_group(f, "fields", "rho", g.rho.shape, link_from="f")
        create_obs_group(f, "fields", "phi", g.phi.shape, link_from="f")

        real_t = 0.0; t_images = 1
        def write(step, write_fields=True):
            compute_rho(g); compute_phi(g)
            mx, my = compute_M(g)
            en_kin = 0.5*np.sum((g.v[None,:]**2)*g.f)*g.dx*g.dv
            en_int = 0.5*(1-mx*mx-my*my)
            mass = np.sum(g.f)*g.dx*g.dv
            momentum = np.sum(g.v[None,:]*g.f)*g.dx*g.dv
            i2 = np.sum(g.f**2)*g.dx*g.dv
            i3 = np.sum(g.f**3)*g.dx*g.dv
            bp = g.dx * g.dv * (np.sum(g.f[:, 0]) + np.sum(g.f[:, -1]))
            vals = dict(mass=mass, energy=en_kin+en_int, en_int=en_int, en_kin=en_kin, momentum=momentum, Mx=mx, My=my, I2=i2, I3=i3, f_min=np.min(g.f), Bp=bp)
            for k,v in vals.items(): append_obs(f,"observables",k,step,real_t,v)
            if write_fields:
                append_obs(f,"fields","f",step,real_t,to_vmf90_field_layout(g.f))
                append_obs(f,"fields","rho",step,real_t,g.rho)
                append_obs(f,"fields","phi",step,real_t,g.phi)
            return mx,my

        mx,my = write(0, write_fields=True)
        print(
            f"[progress] top=0/{n_top} time={real_t:.6f} "
            f"M=({mx:.6e}, {my:.6e}) |M|={np.hypot(mx, my):.6e}",
            flush=True,
        )
        for top in range(1, n_top+1):
            adv_x(g, 0.5)
            for _ in range(n_steps-1):
                compute_rho(g); mx,my = compute_M(g); compute_force(g,mx,my)
                adv_v(g,1.0); adv_x(g,1.0); real_t += dt
            compute_rho(g); mx,my = compute_M(g); compute_force(g,mx,my)
            adv_v(g,1.0); adv_x(g,0.5); real_t += dt
            write_fields = top*n_images//n_top >= t_images
            mx, my = write(top, write_fields=write_fields)
            if (top % log_every) == 0 or top == n_top:
                print(
                    f"[progress] top={top}/{n_top} time={real_t:.6f} "
                    f"M=({mx:.6e}, {my:.6e}) |M|={np.hypot(mx, my):.6e} "
                    f"fields_written={write_fields}",
                    flush=True,
                )
            if write_fields:
                t_images += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="HMF_in.toml")
    ap.add_argument("--output", default="hmf.h5")
    args = ap.parse_args()
    run(parse_config(args.config), args.output)
