#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

struct Header {
    char magic[8];
    uint32_t ntheta;
    uint32_t np;
    double pmax;
    double time;
};

static constexpr double PI = 3.1415926535897932384626433832795;

struct Diagnostics {
    double mass, energy, mx, my, m, l2, fmin, bp;
};

static bool load_state(const char* path, Header& h, std::vector<double>& f) {
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    if (std::fread(&h, sizeof(Header), 1, fp) != 1) { std::fclose(fp); return false; }
    if (std::strncmp(h.magic, "HMFV1", 5) != 0) { std::fclose(fp); return false; }
    size_t n = static_cast<size_t>(h.ntheta) * h.np;
    f.resize(n);
    bool ok = std::fread(f.data(), sizeof(double), n, fp) == n;
    std::fclose(fp);
    return ok;
}

static bool save_state(const char* path, const Header& h, const std::vector<double>& f) {
    std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;
    if (std::fwrite(&h, sizeof(Header), 1, fp) != 1) { std::fclose(fp); return false; }
    bool ok = std::fwrite(f.data(), sizeof(double), f.size(), fp) == f.size();
    std::fclose(fp);
    return ok;
}

static inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::fabs(a) < std::fabs(b)) ? a : b;
}

static void build_ppm(const std::vector<double>& q, double left_bc, double right_bc, bool periodic,
                      std::vector<double>& qL, std::vector<double>& qR) {
    int n = (int)q.size();
    qL.resize(n); qR.resize(n);
    std::vector<double> slope(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double qm = periodic ? q[(i - 1 + n) % n] : (i > 0 ? q[i - 1] : left_bc);
        double qp = periodic ? q[(i + 1) % n] : (i + 1 < n ? q[i + 1] : right_bc);
        double dl = q[i] - qm;
        double dr = qp - q[i];
        slope[i] = minmod(0.5 * (qp - qm), 2.0 * minmod(dl, dr));
    }
    for (int i = 0; i < n; ++i) {
        qL[i] = q[i] - 0.5 * slope[i];
        qR[i] = q[i] + 0.5 * slope[i];
        double qmin = std::min({q[i], qL[i], qR[i]});
        if (qmin < 0.0 && q[i] >= 0.0) {
            qL[i] = std::max(0.0, qL[i]);
            qR[i] = std::max(0.0, qR[i]);
        }
        if ((qR[i] - q[i]) * (q[i] - qL[i]) <= 0.0) {
            qL[i] = q[i];
            qR[i] = q[i];
        } else {
            double d2 = 6.0 * (q[i] - 0.5 * (qL[i] + qR[i]));
            if ((qR[i] - qL[i]) * d2 > (qR[i] - qL[i]) * (qR[i] - qL[i])) {
                qL[i] = 3.0 * q[i] - 2.0 * qR[i];
            }
            if ((qR[i] - qL[i]) * d2 < -(qR[i] - qL[i]) * (qR[i] - qL[i])) {
                qR[i] = 3.0 * q[i] - 2.0 * qL[i];
            }
        }
    }
}

static inline double primitive_in_cell(double qbar, double ql, double qr, double xi) {
    // xi in [-0.5, 0.5], return int_{-0.5}^{xi} q(s) ds
    double b = qr - ql;
    double a = 3.0 * (ql + qr - 2.0 * qbar);
    double c = qbar - a / 12.0;
    double x0 = -0.5;
    auto antider = [&](double x) {
        return (a / 3.0) * x * x * x + 0.5 * b * x * x + c * x;
    };
    return antider(xi) - antider(x0);
}

static double integrate_nonperiodic(const std::vector<double>& q, const std::vector<double>& qL,
                                    const std::vector<double>& qR, double dx,
                                    double x0, double x1, double xmin, double xmax) {
    if (x1 <= xmin || x0 >= xmax || x1 <= x0) return 0.0;
    x0 = std::max(x0, xmin);
    x1 = std::min(x1, xmax);
    if (x1 <= x0) return 0.0;
    int n = (int)q.size();
    double sum = 0.0;
    int i0 = std::max(0, (int)std::floor((x0 - xmin) / dx));
    int i1 = std::min(n - 1, (int)std::floor((x1 - xmin - 1e-15) / dx));
    for (int i = i0; i <= i1; ++i) {
        double cl = xmin + i * dx;
        double cr = cl + dx;
        double a = std::max(x0, cl);
        double b = std::min(x1, cr);
        if (b <= a) continue;
        double xil = (a - (cl + 0.5 * dx)) / dx;
        double xir = (b - (cl + 0.5 * dx)) / dx;
        sum += dx * (primitive_in_cell(q[i], qL[i], qR[i], xir) - primitive_in_cell(q[i], qL[i], qR[i], xil));
    }
    return sum;
}

static double integrate_periodic(const std::vector<double>& q, const std::vector<double>& qL,
                                 const std::vector<double>& qR, double dx,
                                 double x0, double x1, double xmin, double xmax, double mass) {
    double L = xmax - xmin;
    if (x1 <= x0) return 0.0;
    auto F = [&](double x)->double {
        double y = (x - xmin) / L;
        double m = std::floor(y);
        double r = x - m * L;
        return m * mass + integrate_nonperiodic(q, qL, qR, dx, xmin, r, xmin, xmax);
    };
    return F(x1) - F(x0);
}

static void advect_1d(const std::vector<double>& in, std::vector<double>& out,
                      double dx, double shift, bool periodic,
                      double xmin, double xmax) {
    int n = (int)in.size();
    std::vector<double> qL, qR;
    build_ppm(in, 0.0, 0.0, periodic, qL, qR);
    double mass = 0.0;
    for (double v : in) mass += v * dx;
    out.assign(n, 0.0);
    for (int k = 0; k < n; ++k) {
        double xl = xmin + k * dx;
        double xr = xl + dx;
        double a = xl - shift;
        double b = xr - shift;
        double integ = periodic
            ? integrate_periodic(in, qL, qR, dx, a, b, xmin, xmax, mass)
            : integrate_nonperiodic(in, qL, qR, dx, a, b, xmin, xmax);
        out[k] = integ / dx;
    }
}

static void theta_advect(const std::vector<double>& f, std::vector<double>& fout,
                         int nt, int np, double dtheta, double dtau, double pmax) {
    double dp = 2.0 * pmax / np;
    fout.assign(f.size(), 0.0);
    std::vector<double> row(np), row2(np), th(nt), th2(nt);
    (void)row; (void)row2;
    for (int j = 0; j < np; ++j) {
        double pj = -pmax + (j + 0.5) * dp;
        for (int i = 0; i < nt; ++i) th[i] = f[i * np + j];
        advect_1d(th, th2, dtheta, pj * dtau, true, 0.0, 2.0 * PI);
        for (int i = 0; i < nt; ++i) fout[i * np + j] = th2[i];
    }
}

static void force_from_f(const std::vector<double>& f, int nt, int np, double dtheta, double pmax,
                         double& mx, double& my, std::vector<double>& force) {
    double dp = 2.0 * pmax / np;
    mx = 0.0; my = 0.0;
    for (int i = 0; i < nt; ++i) {
        double th = (i + 0.5) * dtheta;
        double c = std::cos(th), s = std::sin(th);
        for (int j = 0; j < np; ++j) {
            double val = f[i * np + j] * dtheta * dp;
            mx += val * c;
            my += val * s;
        }
    }
    force.resize(nt);
    for (int i = 0; i < nt; ++i) {
        double th = (i + 0.5) * dtheta;
        force[i] = -mx * std::sin(th) + my * std::cos(th);
    }
}

static void p_advect(const std::vector<double>& f, std::vector<double>& fout,
                     int nt, int np, double dtheta, double dt, double pmax,
                     const std::vector<double>& force) {
    (void)dtheta;
    double dp = 2.0 * pmax / np;
    fout.assign(f.size(), 0.0);
    std::vector<double> p(np), p2(np);
    for (int i = 0; i < nt; ++i) {
        for (int j = 0; j < np; ++j) p[j] = f[i * np + j];
        advect_1d(p, p2, dp, force[i] * dt, false, -pmax, pmax);
        for (int j = 0; j < np; ++j) fout[i * np + j] = p2[j];
    }
}

static Diagnostics compute_diag(const std::vector<double>& f, int nt, int np, double dtheta, double pmax) {
    double dp = 2.0 * pmax / np;
    Diagnostics d{};
    d.fmin = 1e300;
    for (int i = 0; i < nt; ++i) {
        double th = (i + 0.5) * dtheta;
        double c = std::cos(th), s = std::sin(th);
        for (int j = 0; j < np; ++j) {
            double pj = -pmax + (j + 0.5) * dp;
            double v = f[i * np + j];
            double w = dtheta * dp;
            d.mass += v * w;
            d.mx += v * w * c;
            d.my += v * w * s;
            d.l2 += v * v * w;
            d.energy += 0.5 * pj * pj * v * w;
            d.fmin = std::min(d.fmin, v);
            if (j < 2 || j >= np - 2) d.bp += v * w;
        }
    }
    d.energy += 0.5 * (1.0 - d.mx * d.mx - d.my * d.my);
    d.m = std::sqrt(d.mx * d.mx + d.my * d.my);
    return d;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: %s input.bin output.bin dt nsteps\n", argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    double dt = std::atof(argv[3]);
    int nsteps = std::atoi(argv[4]);

    Header h{};
    std::vector<double> f, f1, f2;
    if (!load_state(in_path, h, f)) {
        std::fprintf(stderr, "Failed to load input file: %s\n", in_path);
        return 2;
    }

    int nt = (int)h.ntheta, np = (int)h.np;
    double dtheta = 2.0 * PI / nt;

    Diagnostics d0 = compute_diag(f, nt, np, dtheta, h.pmax);
    std::printf("Initial: mass=%.16e energy=%.16e M=%.16e L2=%.16e fmin=%.16e bp=%.16e\n",
                d0.mass, d0.energy, d0.m, d0.l2, d0.fmin, d0.bp);

    for (int n = 0; n < nsteps; ++n) {
        theta_advect(f, f1, nt, np, dtheta, 0.5 * dt, h.pmax);
        double mx, my;
        std::vector<double> force;
        force_from_f(f1, nt, np, dtheta, h.pmax, mx, my, force);
        p_advect(f1, f2, nt, np, dtheta, dt, h.pmax, force);
        theta_advect(f2, f, nt, np, dtheta, 0.5 * dt, h.pmax);
        h.time += dt;
    }

    Diagnostics d1 = compute_diag(f, nt, np, dtheta, h.pmax);
    std::printf("Final:   mass=%.16e energy=%.16e M=%.16e L2=%.16e fmin=%.16e bp=%.16e\n",
                d1.mass, d1.energy, d1.m, d1.l2, d1.fmin, d1.bp);

    if (!save_state(out_path, h, f)) {
        std::fprintf(stderr, "Failed to save output file: %s\n", out_path);
        return 3;
    }
    return 0;
}
