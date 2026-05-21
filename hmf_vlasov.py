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
        self.dv = (self.vmax - self.vmin) / (self.Nv - 1)
        self.x = self.xmin + np.arange(self.Nx) * self.dx
        self.v = self.vmin + np.arange(self.Nv) * self.dv
        self.f = np.zeros((self.Nx, self.Nv))
        self.rho = np.zeros(self.Nx)
        self.phi = np.zeros(self.Nv)
        self.force = np.zeros(self.Nx)


def init_waterbag(g: Grid, width: float, bag: float):
    mask_x = np.abs(g.x) <= width
    mask_v = np.abs(g.v) <= bag
    g.f[np.ix_(mask_x, mask_v)] = 1.0 / (4 * width * bag)


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


def advance_x(g: Grid, h: float):
    x_back = g.x[:, None] - g.DT * h * g.v[None, :]
    period = g.xmax - g.xmin
    x_eval = ((x_back - g.xmin) % period) + g.xmin

    x_periodic = np.concatenate((g.x, [g.xmax]))

    fnew = np.empty_like(g.f)
    for j in range(g.Nv):
        y_periodic = np.concatenate((g.f[:, j], [g.f[0, j]]))
        spline = CubicSpline(x_periodic, y_periodic, bc_type="periodic", extrapolate="periodic")
        fnew[:, j] = spline(x_eval[:, j])
    g.f = fnew


def advance_v(g: Grid, h: float):
    v_back = g.v[None, :] - g.DT * h * g.force[:, None]

    fnew = np.zeros_like(g.f)
    for i in range(g.Nx):
        spline = CubicSpline(g.v, g.f[i, :], bc_type="natural", extrapolate=False)
        vals = spline(v_back[i, :])
        vals[np.isnan(vals)] = 0.0
        fnew[i, :] = vals
    g.f = fnew


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


def run(conf, out_file="hmf.h5"):
    nx, nv = int(conf["Nx"]), int(conf["Nv"])
    vmax = float(conf["vmax"])
    dt = float(conf["DT"])
    g = Grid(nx, nv, -PI, PI, -vmax, vmax, dt)
    init_waterbag(g, float(conf["width"]), float(conf["bag"]))

    n_steps, n_top = int(conf["n_steps"]), int(conf["n_top"])
    n_images = int(conf.get("n_images", n_top))

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

        for k, v in {"model":"HMF","Nx":nx,"Nv":nv,"DT":dt,"xmin":g.xmin,"xmax":g.xmax,"vmin":g.vmin,"vmax":g.vmax,"dx":g.dx,"dv":g.dv}.items():
            f[f"parameters/{k}"] = v

        obs_scalars = ["mass","energy","en_int","en_kin","momentum","Mx","My","I2","I3"]
        create_obs_group(f, "observables", "mass", ())
        for n in obs_scalars[1:]: create_obs_group(f, "observables", n, (), link_from="mass")
        create_obs_group(f, "fields", "f", g.f.shape)
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
            vals = dict(mass=mass, energy=en_kin+en_int, en_int=en_int, en_kin=en_kin, momentum=momentum, Mx=mx, My=my, I2=i2, I3=i3)
            for k,v in vals.items(): append_obs(f,"observables",k,step,real_t,v)
            if write_fields:
                append_obs(f,"fields","f",step,real_t,g.f)
                append_obs(f,"fields","rho",step,real_t,g.rho)
                append_obs(f,"fields","phi",step,real_t,g.phi)
            return mx,my

        mx,my = write(0, write_fields=True)
        for top in range(1, n_top+1):
            advance_x(g, 0.5)
            for _ in range(n_steps-1):
                compute_rho(g); mx,my = compute_M(g); compute_force(g,mx,my)
                advance_v(g,1.0); advance_x(g,1.0); real_t += dt
            compute_rho(g); mx,my = compute_M(g); compute_force(g,mx,my)
            advance_v(g,1.0); advance_x(g,0.5); real_t += dt
            write_fields = top*n_images//n_top >= t_images
            write(top, write_fields=write_fields)
            if write_fields:
                t_images += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="HMF_in")
    ap.add_argument("--output", default="hmf.h5")
    args = ap.parse_args()
    run(parse_config(args.config), args.output)
