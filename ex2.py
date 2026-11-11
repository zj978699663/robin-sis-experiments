import csv
import math
from math import sqrt
from statistics import median

import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.bdf import BDF
from scipy.linalg.lapack import dgbtrf, dgbtrs
from threadpoolctl import threadpool_limits

from common import (
    DATA,
    REFERENCE,
    TABLES,
    Basis,
    FourierGalerkin,
    P1Mesh,
    Parameters,
    save_table,
    steady_and_derivative,
)


EPS = (0.04, 0.01, 0.0025)
LEVELS = (10, 20, 40, 80, 160, 320)
METHODS = ("enriched", "plain", "direct")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        rows = comparison()
    save_table(
        rows,
        REFERENCE / "table3.csv",
        TABLES / "table3.csv",
        (None, None, 2, 2, 2, 2, 2, 2),
    )
    print("Table 3 checked")


def historical_times(path):
    groups = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = float(row["epsilon"]), int(row["ne"]), row["method"]
            groups.setdefault(key, {})[int(row["repetition"])] = float(row["seconds"])
    times = {}
    for eps in EPS:
        for ne in LEVELS:
            for method in METHODS:
                trials = groups[eps, ne, method]
                assert set(trials) == {1, 2, 3}
                times[eps, ne, method] = median(trials.values())
    return times


def comparison():
    par = Parameters()
    x, w = field_rule()
    times = historical_times(DATA / "timings.csv")
    steady = {ne: steady_and_derivative(ne, par) for ne in LEVELS}
    rows = []
    for eps in EPS:
        target = reference(eps, par, x)
        print(f"Reference ready for epsilon={eps:g}", flush=True)
        for ne in LEVELS:
            errors = case_errors(ne, eps, par, steady[ne], target, x, w)
            rows.append(table_row(eps, ne, errors, times))
            print(
                f"epsilon={eps:g}, h=1/{ne}: " + ", ".join(f"{e:.4g}" for e in errors),
                flush=True,
            )
    return rows


def table_row(eps, ne, errors, times):
    row = [eps, f"1/{ne}"]
    for method, error in zip(METHODS, errors, strict=True):
        row.extend((error, times[eps, ne, method]))
    return row


def case_errors(ne, eps, par, steady, target, x, w):
    errors = []
    for enriched in (True, False):
        model = FourierGalerkin(ne, 3, eps, par, phase_points=80, enriched=enriched)
        coef = solve_fourier(model, model.initial(*steady))
        field = mode_values(model, coef, x)
        total = 0.0
        for values, ref in zip(field, target, strict=True):
            d = -ref.copy()
            d[: len(values)] += values
            total += float(
                np.real(np.sum(w * abs(d[0]) ** 2) + 2 * np.sum(w * abs(d[1:]) ** 2))
            )
        errors.append(sqrt(max(total, 0)))
    fields = cold_march(ne, eps, steady[0], par)
    errors.append(direct_error(fields, target, x, w))
    return errors


def field_rule():
    xi, wi = leggauss(6)
    x = ((np.arange(10240)[:, None] + (xi + 1) / 2) / 10240).ravel()
    return x, np.tile(wi / (2 * 10240), 10240)


def reference(eps, par, x):
    coarse = FourierGalerkin(5120, 5, eps, par, phase_points=112)
    steady = steady_and_derivative(coarse.mesh.ne, par)
    c0 = solve_fourier(coarse, coarse.initial(*steady))
    fine = FourierGalerkin(10240, 5, eps, par, phase_points=112)
    c1 = solve_fourier(fine, nested_initial(coarse, c0, fine))
    return (4 * mode_values(fine, c1, x) - mode_values(coarse, c0, x)) / 3


def solve_fourier(model, initial):
    coef, failure = model.solve_newton_gmres(initial, 1e-10, max_newton=20)
    if failure is not None:
        raise RuntimeError(f"epsilon={model.eps:g}, ne={model.mesh.ne}: {failure}")
    Sq, Iq = model.reconstruct_quadrature(*model.unpack(coef))
    if min(Sq.min(), Iq.min()) <= 0:
        raise RuntimeError(
            f"epsilon={model.eps:g}, ne={model.mesh.ne}: nonpositive solution"
        )
    return coef


def mode_values(model, coef, x):
    S, I = model.unpack(coef)
    nn = model.mesh.nn
    Sx = [np.interp(x, model.mesh.nodes, S[k]) for k in range(model.N + 1)]
    Ix = []
    for k in range(model.N + 1):
        basis = model.ib[k]
        ordinary = I[k][:nn].copy()
        amps = I[k][nn:] / basis.scales
        for amp, proj in zip(amps, basis.projections, strict=True):
            ordinary -= amp * proj
        values = np.interp(x, model.mesh.nodes, ordinary)
        if len(amps):
            raw, _, _ = Basis._raw_layers(x, k, model.eps, model.par.dI)
            for amp, layer in zip(amps, raw, strict=True):
                values += amp * layer
        Ix.append(values)
    return np.array((Sx, Ix))


def direct_error(fields, target, x, w):
    nodes = fields["x"]
    h = np.diff(nodes)
    total = 0.0
    for name, ref in zip(("S", "I"), target, strict=True):
        u = fields[name]
        modes = np.fft.rfft(u, axis=0) / len(u)
        for k in range(len(ref)):
            d = np.interp(x, nodes, modes[k]) - ref[k]
            total += (1 if k == 0 else 2) * float(np.sum(w * abs(d) ** 2))
        for k in range(len(ref), len(modes)):
            a, b = modes[k, :-1], modes[k, 1:]
            weight = 1 if k == len(modes) - 1 else 2
            total += weight * float(
                np.real(
                    np.sum(
                        h * (abs(a) ** 2 + np.real(a * np.conj(b)) + abs(b) ** 2) / 3
                    )
                )
            )
    return sqrt(max(total, 0))


def cold_march(ne, eps, steady, par):
    model = FullP1(ne, eps, par)
    state = np.column_stack((steady[: ne + 1], steady[ne + 1 :])).ravel()
    cycles, defect = 0, float("inf")
    tol, rtol, atol, max_cycles = 2e-12, 2e-12, 2e-14, 20000

    def advance(a, b, u, rt, at, step, **kwargs):
        sol = model.integrate(a, b, u, rt, at, step, **kwargs)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    while cycles < max_cycles:
        end = min(cycles + 100, max_cycles)
        sol = advance(float(cycles), float(end), state, 2e-7, 2e-9, 0.05)
        state, cycles = sol.y[:, -1], end
        one = advance(float(cycles), float(cycles + 1), state, 2e-8, 2e-10, 0.025)
        defect = float(np.linalg.norm(one.y[:, -1] - state) / np.linalg.norm(state))
        if defect < 1e-6:
            state = one.y[:, -1]
            cycles += 1
            break

    while defect >= tol and cycles < max_cycles:
        one = advance(float(cycles), float(cycles + 1), state, rtol, atol, 1 / 80)
        nxt = one.y[:, -1]
        defect = float(np.linalg.norm(nxt - state) / np.linalg.norm(state))
        state = nxt
        cycles += 1
    if defect >= tol:
        raise RuntimeError("Periodic iteration did not converge")

    for _ in range(1000):
        final = advance(
            float(cycles),
            float(cycles + 1),
            state,
            rtol,
            atol,
            1 / 80,
            t_eval=cycles + np.arange(257) / 256,
        )
        defect = float(np.linalg.norm(final.y[:, -1] - state) / np.linalg.norm(state))
        if defect < tol:
            S, I = final.y[0::2, :-1].T, final.y[1::2, :-1].T
            if min(float(S.min()), float(I.min())) <= 0:
                raise RuntimeError("Nonpositive sampled period")
            return {"S": S, "I": I, "x": model.mesh.nodes}
        state = final.y[:, -1]
        cycles += 1
    raise RuntimeError("Sampled period did not reach the stopping tolerance")


def nested_initial(coarse, coarse_coef, fine):
    source_s, source_i = coarse.unpack(coarse_coef)
    target_s, target_i = {}, {}
    for k in range(fine.N + 1):
        target_s[k] = np.interp(fine.mesh.nodes, coarse.mesh.nodes, source_s[k])
        ordinary = source_i[k][: coarse.mesh.nn].copy()
        raw_coefficients = []
        for j, (projection, scale) in enumerate(
            zip(coarse.ib[k].projections, coarse.ib[k].scales)
        ):
            coefficient = source_i[k][coarse.mesh.nn + j] / scale
            ordinary -= coefficient * projection
            raw_coefficients.append(coefficient)
        target_ordinary = np.interp(fine.mesh.nodes, coarse.mesh.nodes, ordinary)
        target_extra = []
        assert len(raw_coefficients) == len(fine.ib[k].scales)
        for coefficient, projection, scale in zip(
            raw_coefficients, fine.ib[k].projections, fine.ib[k].scales
        ):
            target_ordinary += coefficient * projection
            target_extra.append(coefficient * scale)
        target_i[k] = np.concatenate([target_ordinary, np.asarray(target_extra)])
    return fine.pack(target_s, target_i)


@njit
def mass_solve(values, piv, multipliers, off):
    z = values.copy()
    n = len(piv)
    for i in range(1, n):
        for j in range(2):
            z[i, j] -= multipliers[i - 1] * z[i - 1, j]
    for j in range(2):
        z[n - 1, j] /= piv[n - 1]
    for i in range(n - 2, -1, -1):
        for j in range(2):
            z[i, j] = (z[i, j] - off[i] * z[i + 1, j]) / piv[i]
    return z.ravel()


@njit
def assembled_rhs(
    u, left, right, w, kmain, koff, piv, multipliers, moff, params, q, eps
):
    ds, di, a, b, rho, mu, beta, gamma, theta = params
    n = len(kmain)
    U = u.reshape(n, 2)
    weak = np.zeros((n, 2))
    for i in range(n - 1):
        for j in range(left.shape[1]):
            l, r = left[i, j], right[i, j]
            s = l * U[i, 0] + r * U[i + 1, 0]
            v = l * U[i, 1] + r * U[i + 1, 1]
            inc = beta * s * v / max(s + v, 1e-10)
            fs = s * (rho - mu * s) - inc + theta * gamma * v
            fi = inc - gamma * v
            weak[i, 0] += w[i, j] * l * fs
            weak[i + 1, 0] += w[i, j] * r * fs
            weak[i, 1] += w[i, j] * l * fi
            weak[i + 1, 1] += w[i, j] * r * fi
    for i in range(n):
        for j in range(2):
            ku = kmain[i] * U[i, j]
            if i > 0:
                ku += koff[i - 1] * U[i - 1, j]
            if i < n - 1:
                ku += koff[i] * U[i + 1, j]
            d = ds if j == 0 else di
            weak[i, j] -= d * ku
            if i == 0 or i == n - 1:
                weak[i, j] -= d * (a if j == 0 else b + q) * U[i, j]
            weak[i, j] *= eps
    return mass_solve(weak, piv, multipliers, moff)


@njit
def assembled_jac(u, left, right, w, kmain, koff, params, q, eps):
    ds, di, a, b, rho, mu, beta, gamma, theta = params
    n = len(kmain)
    U = u.reshape(n, 2)
    band = np.zeros((7, 2 * n))
    for i in range(n - 1):
        for j in range(left.shape[1]):
            l, r = left[i, j], right[i, j]
            s = l * U[i, 0] + r * U[i + 1, 0]
            v = l * U[i, 1] + r * U[i + 1, 1]
            den = s + v
            if den <= 1e-10:
                raise ValueError("Positive branch required")
            incs = beta * v * v / (den * den)
            inci = beta * s * s / (den * den)
            d = (rho - 2 * mu * s - incs, -inci + theta * gamma, incs, inci - gamma)
            for k in range(4):
                rs, cs = k // 2, k % 2
                for aa in range(2):
                    row = 2 * (i + aa) + rs
                    na = l if aa == 0 else r
                    for bb in range(2):
                        col = 2 * (i + bb) + cs
                        nb = l if bb == 0 else r
                        band[3 + row - col, col] += eps * w[i, j] * d[k] * na * nb
    for j in range(2):
        d = ds if j == 0 else di
        robin = a if j == 0 else b + q
        for i in range(n):
            col = 2 * i + j
            band[3, col] -= eps * d * kmain[i]
            if i == 0 or i == n - 1:
                band[3, col] -= eps * d * robin
        for i in range(n - 1):
            col = 2 * i + j
            band[1, col + 2] -= eps * d * koff[i]
            band[5, col] -= eps * d * koff[i]
    return band


class ScaledJacobian:
    def __init__(self, c, weak):
        self.c, self.weak = c, weak


class MassJacobian:
    def __init__(self, weak):
        self.weak = weak

    def __rmul__(self, c):
        return ScaledJacobian(c, self.weak)


class MassIdentity:
    def __sub__(self, scaled):
        return scaled


class MassBDF(BDF):
    def __init__(self, *args, mass_band=None, mass_apply=None, **kwargs):
        self.mass_band, self.mass_apply = mass_band, mass_apply
        super().__init__(*args, **kwargs)
        self.I = MassIdentity()
        self.lu, self.solve_lu = self.factor, self.solve_mass

    def _validate_jac(self, jac, sparsity):
        def wrapped(t, y):
            self.njev += 1
            return MassJacobian(jac(t, y))

        return wrapped, wrapped(self.t, self.y)

    def factor(self, scaled):
        ab = np.zeros((10, self.n), order="F")
        ab[3:] = self.mass_band - scaled.c * scaled.weak
        lu, piv, info = dgbtrf(ab, 3, 3, overwrite_ab=True)
        if info:
            raise RuntimeError(f"Banded Newton factorization: {info}")
        self.nlu += 1
        return lu, piv

    def solve_mass(self, factors, rhs):
        lu, piv = factors
        answer, info = dgbtrs(lu, 3, 3, self.mass_apply(rhs), piv)
        if info:
            raise RuntimeError(f"Banded Newton solve: {info}")
        return answer


class FullP1:
    def __init__(self, ne, eps, par):
        self.ne, self.eps, self.par = ne, eps, par
        self.mesh = P1Mesh(ne)
        self.nn = ne + 1
        self.mass = self.mesh.M
        self.kmain, self.koff = self.mesh.K.diagonal(), self.mesh.K.diagonal(1)
        local = ((self.mesh.xq - self.mesh.nodes[self.mesh.elem]) * ne).reshape(ne, 12)
        self.right, self.left = local, 1 - local
        self.w = self.mesh.wq.reshape(ne, 12)
        self.params = np.array(
            (
                par.dS,
                par.dI,
                par.a,
                par.bbar,
                par.rho,
                par.mu,
                par.beta,
                par.gamma,
                par.theta,
            )
        )
        self.mpiv = self.mass.diagonal().copy()
        self.moff = self.mass.diagonal(1).copy()
        self.mmult = np.empty(ne)
        for i in range(ne):
            self.mmult[i] = self.moff[i] / self.mpiv[i]
            self.mpiv[i + 1] -= self.mmult[i] * self.moff[i]
        self.mass_band = np.zeros((7, 2 * self.nn))
        for species in (0, 1):
            for offset in (-1, 0, 1):
                self.put_diagonal(
                    self.mass_band, self.mass.diagonal(offset), species, species, offset
                )

    def put_diagonal(self, target, values, row_species, col_species, offset):
        row_nodes = np.arange(max(0, -offset), self.nn - max(0, offset))
        cols = 2 * (row_nodes + offset) + col_species
        rows = 2 * row_nodes + row_species
        target[3 + rows - cols, cols] = values

    def mass_apply(self, u):
        return np.asarray(self.mass @ u.reshape(self.nn, 2)).ravel()

    def q(self, s):
        p = self.par
        return 2 * p.q1 * math.cos(2 * math.pi * s) + 2 * p.q2 * math.cos(
            4 * math.pi * s
        )

    def rhs(self, s, u):
        return assembled_rhs(
            np.ascontiguousarray(u),
            self.left,
            self.right,
            self.w,
            self.kmain,
            self.koff,
            self.mpiv,
            self.mmult,
            self.moff,
            self.params,
            self.q(s),
            self.eps,
        )

    def jac(self, s, u):
        return assembled_jac(
            np.ascontiguousarray(u),
            self.left,
            self.right,
            self.w,
            self.kmain,
            self.koff,
            self.params,
            self.q(s),
            self.eps,
        )

    def integrate(self, start, end, state, rtol, atol, max_step, **kwargs):
        return solve_ivp(
            self.rhs,
            (start, end),
            state,
            method=MassBDF,
            jac=self.jac,
            mass_band=self.mass_band,
            mass_apply=self.mass_apply,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            **kwargs,
        )


if __name__ == "__main__":
    main()
