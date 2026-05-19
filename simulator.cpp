#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

struct SimParams {
    int ntheta;
    int np;
    double pmax;
    double dt;
    int nsteps;
};

static inline int idx(int i, int j, int np) { return i * np + j; }

static void compute_theta_grid(int ntheta, std::vector<double>& theta) {
    const double dtheta = 2.0 * M_PI / static_cast<double>(ntheta);
    theta.resize(ntheta);
    for (int i = 0; i < ntheta; ++i) theta[i] = (i + 0.5) * dtheta;
}

static double integrate_periodic_linear(const std::vector<double>& row, double left, double right, double dx) {
    const int n = static_cast<int>(row.size());
    const double L = n * dx;
    while (left < 0.0) { left += L; right += L; }
    while (left >= L) { left -= L; right -= L; }

    auto val = [&](double x)->double {
        x = std::fmod(x, L);
        if (x < 0) x += L;
        double z = x / dx;
        int i0 = static_cast<int>(std::floor(z));
        double t = z - i0;
        int i1 = (i0 + 1) % n;
        return (1.0 - t) * row[i0] + t * row[i1];
    };

    double sum = 0.0;
    double x = left;
    while (x < right) {
        double xn = std::min(right, (std::floor(x / dx) + 1.0) * dx);
        const double xm = 0.5 * (x + xn);
        sum += val(xm) * (xn - x);
        x = xn;
    }
    return sum;
}

static double integrate_zero_linear(const std::vector<double>& col, double left, double right, double dx, double x0) {
    const int n = static_cast<int>(col.size());
    const double xmax = x0 + n * dx;
    if (right <= x0 || left >= xmax) return 0.0;
    left = std::max(left, x0);
    right = std::min(right, xmax);

    auto val = [&](double x)->double {
        if (x < x0 || x >= xmax) return 0.0;
        double z = (x - x0) / dx;
        int j0 = static_cast<int>(std::floor(z));
        if (j0 < 0) return 0.0;
        if (j0 >= n - 1) return col[n - 1];
        double t = z - j0;
        return (1.0 - t) * col[j0] + t * col[j0 + 1];
    };

    double sum = 0.0;
    double x = left;
    while (x < right) {
        double cell = x0 + (std::floor((x - x0) / dx) + 1.0) * dx;
        double xn = std::min(right, cell);
        const double xm = 0.5 * (x + xn);
        sum += val(xm) * (xn - x);
        x = xn;
    }
    return sum;
}

static void advect_theta(std::vector<double>& f, const SimParams& p, double tau, double dtheta, const std::vector<double>& pgrid) {
    std::vector<double> out(f.size(), 0.0);
    std::vector<double> row(p.ntheta);
    for (int j = 0; j < p.np; ++j) {
        for (int i = 0; i < p.ntheta; ++i) row[i] = f[idx(i, j, p.np)];
        const double d = pgrid[j] * tau;
        for (int i = 0; i < p.ntheta; ++i) {
            double xl = i * dtheta - d;
            double xr = (i + 1) * dtheta - d;
            out[idx(i, j, p.np)] = integrate_periodic_linear(row, xl, xr, dtheta) / dtheta;
        }
    }
    f.swap(out);
}

static void compute_magnetization(const std::vector<double>& f, const SimParams& p, const std::vector<double>& theta,
                                  double dtheta, double dp, double& mx, double& my) {
    mx = 0.0; my = 0.0;
    for (int i = 0; i < p.ntheta; ++i) {
        const double c = std::cos(theta[i]);
        const double s = std::sin(theta[i]);
        for (int j = 0; j < p.np; ++j) {
            double v = f[idx(i, j, p.np)];
            mx += v * c;
            my += v * s;
        }
    }
    mx *= dtheta * dp;
    my *= dtheta * dp;
}

static void advect_p(std::vector<double>& f, const SimParams& p, double tau, double dp, const std::vector<double>& theta,
                     double p0, double mx, double my) {
    std::vector<double> out(f.size(), 0.0);
    std::vector<double> col(p.np);
    for (int i = 0; i < p.ntheta; ++i) {
        const double force = -mx * std::sin(theta[i]) + my * std::cos(theta[i]);
        for (int j = 0; j < p.np; ++j) col[j] = f[idx(i, j, p.np)];
        const double d = force * tau;
        for (int j = 0; j < p.np; ++j) {
            double xl = p0 + j * dp - d;
            double xr = p0 + (j + 1) * dp - d;
            out[idx(i, j, p.np)] = integrate_zero_linear(col, xl, xr, dp, p0) / dp;
        }
    }
    f.swap(out);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::printf("Usage: %s ntheta np pmax dt nsteps input.txt output.txt\n", argv[0]);
        return 1;
    }

    SimParams p;
    p.ntheta = std::atoi(argv[1]);
    p.np = std::atoi(argv[2]);
    p.pmax = std::atof(argv[3]);
    p.dt = std::atof(argv[4]);
    p.nsteps = std::atoi(argv[5]);
    const char* infile = argv[6];
    const char* outfile = argv[7];

    if (p.ntheta <= 0 || p.np <= 0 || p.pmax <= 0.0 || p.dt <= 0.0 || p.nsteps < 0) {
        std::printf("Invalid simulation parameters.\n");
        return 1;
    }

    const double dtheta = 2.0 * M_PI / static_cast<double>(p.ntheta);
    const double dp = 2.0 * p.pmax / static_cast<double>(p.np);
    const double p0 = -p.pmax;

    std::vector<double> theta;
    compute_theta_grid(p.ntheta, theta);
    std::vector<double> pgrid(p.np);
    for (int j = 0; j < p.np; ++j) pgrid[j] = p0 + (j + 0.5) * dp;

    std::vector<double> f(static_cast<size_t>(p.ntheta) * static_cast<size_t>(p.np), 0.0);
    FILE* fin = std::fopen(infile, "r");
    if (!fin) {
        std::printf("Failed to open input file: %s\n", infile);
        return 1;
    }

    for (int i = 0; i < p.ntheta; ++i) {
        for (int j = 0; j < p.np; ++j) {
            if (std::fscanf(fin, "%lf", &f[idx(i, j, p.np)]) != 1) {
                std::printf("Input format error while reading cell (%d,%d).\n", i, j);
                std::fclose(fin);
                return 1;
            }
        }
    }
    std::fclose(fin);

    for (int n = 0; n < p.nsteps; ++n) {
        advect_theta(f, p, 0.5 * p.dt, dtheta, pgrid);
        double mx, my;
        compute_magnetization(f, p, theta, dtheta, dp, mx, my);
        advect_p(f, p, p.dt, dp, theta, p0, mx, my);
        advect_theta(f, p, 0.5 * p.dt, dtheta, pgrid);
    }

    FILE* fout = std::fopen(outfile, "w");
    if (!fout) {
        std::printf("Failed to open output file: %s\n", outfile);
        return 1;
    }

    for (int i = 0; i < p.ntheta; ++i) {
        for (int j = 0; j < p.np; ++j) {
            std::fprintf(fout, "%.17g%c", f[idx(i, j, p.np)], (j + 1 == p.np) ? '\n' : ' ');
        }
    }

    std::fclose(fout);
    std::printf("Simulation complete. Wrote final distribution to %s\n", outfile);
    return 0;
}
