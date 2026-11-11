from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags, hstack
from scipy.sparse.linalg import LinearOperator, gmres, splu


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
REFERENCE = ROOT / "reference"
DATA = ROOT / "data"
TABLES = BUILD / "tables"


def save_table(rows, reference, path, digits):
    with reference.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        expected = list(reader)

    printed = []
    for i, (row, ref) in enumerate(zip(rows, expected, strict=True), 1):
        cells = [
            str(x) if n is None else f"{x:.{n}g}"
            for x, n in zip(row, digits, strict=True)
        ]
        for name, value, target in zip(header, cells, ref, strict=True):
            if value != target and (name == "h" or Decimal(value) != Decimal(target)):
                raise ValueError(
                    f"{path.name}, row {i}, {name}: {value} versus {target}"
                )
        printed.append(cells)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(printed)


@dataclass(frozen=True)
class Parameters:
    dS: float = 0.08
    dI: float = 0.05
    a: float = 0.12
    bbar: float = 0.50
    rho: float = 2.0
    mu: float = 1.0
    beta: float = 3.0
    gamma: float = 1.0
    theta: float = 0.45
    q1: float = 0.22
    q2: float = 0.14

    @property
    def K(self) -> int:
        return 2

    def q(self, k: int) -> complex:
        if abs(k) == 1:
            return self.q1
        if abs(k) == 2:
            return self.q2
        return 0.0

    def kappa(self, N: int | None = None) -> float:
        top = self.K if N is None else min(N, self.K)
        return sum(
            abs(self.q(k)) ** 2 * math.sqrt(self.dI / (math.pi * k))
            for k in range(1, top + 1)
        )


class P1Mesh:
    def __init__(self, ne: int, ngauss: int = 12):
        self.ne = ne
        self.nn = ne + 1
        self.nodes = np.linspace(0.0, 1.0, self.nn)
        xi, wi = leggauss(ngauss)
        xs, ws, elem = [], [], []
        h = 1.0 / ne
        for e in range(ne):
            left = self.nodes[e]
            xs.append(left + 0.5 * h * (xi + 1.0))
            ws.append(0.5 * h * wi)
            elem.extend([e] * ngauss)
        self.xq = np.concatenate(xs)
        self.wq = np.concatenate(ws)
        self.elem = np.asarray(elem, dtype=int)
        nq = len(self.xq)
        local = (self.xq - self.nodes[self.elem]) / h
        rows = np.arange(nq)
        row_index = np.concatenate([rows, rows])
        col_index = np.concatenate([self.elem, self.elem + 1])
        self.phi = csr_matrix(
            (np.concatenate([1.0 - local, local]), (row_index, col_index)),
            shape=(nq, self.nn),
        )
        self.dphi = csr_matrix(
            (
                np.concatenate([-np.ones(nq) / h, np.ones(nq) / h]),
                (row_index, col_index),
            ),
            shape=(nq, self.nn),
        )
        self.ep = csr_matrix(
            (np.ones(2), (np.arange(2), np.array([0, self.nn - 1]))),
            shape=(2, self.nn),
        )
        weight = diags(self.wq, format="csc")
        self.M = csc_matrix(self.phi.T @ weight @ self.phi)
        self.K = csc_matrix(self.dphi.T @ weight @ self.dphi)
        self.H = csc_matrix(self.M + self.K)

    def p1_eval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        h = 1.0 / self.ne
        e = np.minimum((x / h).astype(int), self.ne - 1)
        local = (x - self.nodes[e]) / h
        val = np.zeros((len(x), self.nn))
        der = np.zeros_like(val)
        rows = np.arange(len(x))
        val[rows, e] = 1.0 - local
        val[rows, e + 1] = local
        der[rows, e] = -1.0 / h
        der[rows, e + 1] = 1.0 / h
        return val, der


class Basis:
    def __init__(
        self, mesh: P1Mesh, k: int, eps: float, dI: float, enriched: bool = True
    ):
        self.mesh = mesh
        self.k = k
        self.eps = eps
        self.dI = dI
        self.projections: list[np.ndarray] = []
        self.scales: list[float] = []
        if k == 0 or not enriched:
            self.val = mesh.phi.astype(complex)
            self.der = mesh.dphi.astype(complex)
            self.ep = mesh.ep.astype(complex)
        else:
            raw_cols, raw_der_cols, raw_ep_cols = self._raw_layers(mesh.xq, k, eps, dI)
            cols = [mesh.phi.astype(complex)]
            dcols = [mesh.dphi.astype(complex)]
            epcols = [mesh.ep.astype(complex)]
            h_lu = splu(mesh.H.astype(complex))
            for raw, draw, rep in zip(raw_cols, raw_der_cols, raw_ep_cols):
                rhs = mesh.phi.T @ (mesh.wq * raw) + mesh.dphi.T @ (mesh.wq * draw)
                proj = h_lu.solve(np.asarray(rhs).ravel())
                z = raw - mesh.phi @ proj
                dz = draw - mesh.dphi @ proj
                zep = rep - mesh.ep @ proj

                norm = math.sqrt(
                    float(np.real(np.vdot(z, mesh.wq * z) + np.vdot(dz, mesh.wq * dz)))
                )
                if norm < 1.0e-11:
                    continue
                self.projections.append(proj)
                self.scales.append(norm)
                cols.append(csc_matrix((z / norm)[:, None]))
                dcols.append(csc_matrix((dz / norm)[:, None]))
                epcols.append(csc_matrix((zep / norm)[:, None]))
            self.val = hstack(cols, format="csr")
            self.der = hstack(dcols, format="csr")
            self.ep = hstack(epcols, format="csr")
        self.n = self.val.shape[1]
        weight = diags(mesh.wq, format="csc")
        self.M = csc_matrix(self.val.conj().T @ weight @ self.val)
        self.K = csc_matrix(self.der.conj().T @ weight @ self.der)
        self.B = csc_matrix(self.ep.conj().T @ self.ep)

    @staticmethod
    def _cutoff(r: np.ndarray, L: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
        y = r / L
        c = np.zeros_like(r)
        dc = np.zeros_like(r)
        m = y < 1.0
        c[m] = 1.0 - 3.0 * y[m] ** 2 + 2.0 * y[m] ** 3
        dc[m] = (-6.0 * y[m] + 6.0 * y[m] ** 2) / L
        return c, dc

    @classmethod
    def _raw_layers(cls, x: np.ndarray, k: int, eps: float, dI: float):
        mu = np.sqrt(1j * 2.0 * np.pi * k / (dI * eps))
        if mu.real < 0:
            mu = -mu
        result_v, result_d, result_ep = [], [], []
        for side in (0, 1):
            r = x if side == 0 else 1.0 - x
            sign = 1.0 if side == 0 else -1.0
            c, dcdr = cls._cutoff(r)
            raw = np.exp(-mu * r) * c
            dr = np.exp(-mu * r) * (dcdr - mu * c) * sign
            rep = np.array([1.0 + 0j, 0j]) if side == 0 else np.array([0j, 1.0 + 0j])
            result_v.append(raw)
            result_d.append(dr)
            result_ep.append(rep)
        return result_v, result_d, result_ep

    def eval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p, dp = self.mesh.p1_eval(x)
        if self.k == 0 or not self.projections:
            return p.astype(complex), dp.astype(complex)
        raw_cols, raw_der_cols, _ = self._raw_layers(x, self.k, self.eps, self.dI)
        cols = [p.astype(complex)]
        dcols = [dp.astype(complex)]
        for raw, draw, proj, scale in zip(
            raw_cols, raw_der_cols, self.projections, self.scales
        ):
            cols.append(((raw - p @ proj) / scale)[:, None])
            dcols.append(((draw - dp @ proj) / scale)[:, None])
        return np.hstack(cols), np.hstack(dcols)


class FourierGalerkin:
    def __init__(
        self,
        ne: int,
        N: int,
        eps: float,
        par: Parameters,
        phase_points: int | None = None,
        enriched: bool = True,
        volume_points: int = 12,
    ):
        self.ne, self.N, self.eps, self.par = ne, N, eps, par
        self.enriched = enriched
        self.mesh = P1Mesh(ne, ngauss=volume_points)
        self.sb = Basis(self.mesh, 0, eps, par.dI, enriched=False)
        self.ib = {0: Basis(self.mesh, 0, eps, par.dI, enriched=False)}
        for k in range(1, N + 1):
            self.ib[k] = Basis(self.mesh, k, eps, par.dI, enriched=enriched)
        self.Q = phase_points or max(64, 16 * N + 32)
        self.s = np.arange(self.Q) / self.Q
        self.exp_pos = {
            k: np.exp(1j * 2.0 * np.pi * k * self.s) for k in range(1, N + 1)
        }
        self.slices: dict[tuple[int, str], slice] = {}
        cur = 0
        self.slices[(0, "S")] = slice(cur, cur + self.mesh.nn)
        cur += self.mesh.nn
        self.slices[(0, "I")] = slice(cur, cur + self.mesh.nn)
        cur += self.mesh.nn
        for k in range(1, N + 1):
            for species, size in (("S", self.mesh.nn), ("I", self.ib[k].n)):
                self.slices[(k, species, "re")] = slice(cur, cur + size)
                cur += size
                self.slices[(k, species, "im")] = slice(cur, cur + size)
                cur += size
        self.ndof = cur

    def unpack(
        self, x: np.ndarray
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        S = {0: x[self.slices[(0, "S")]].astype(complex)}
        I = {0: x[self.slices[(0, "I")]].astype(complex)}
        for k in range(1, self.N + 1):
            S[k] = x[self.slices[(k, "S", "re")]] + 1j * x[self.slices[(k, "S", "im")]]
            I[k] = x[self.slices[(k, "I", "re")]] + 1j * x[self.slices[(k, "I", "im")]]
        return S, I

    def pack(self, S: dict[int, np.ndarray], I: dict[int, np.ndarray]) -> np.ndarray:
        x = np.zeros(self.ndof)
        x[self.slices[(0, "S")]] = np.real(S[0])
        x[self.slices[(0, "I")]] = np.real(I[0])
        for k in range(1, self.N + 1):
            for species, data in (("S", S[k]), ("I", I[k])):
                x[self.slices[(k, species, "re")]] = np.real(data)
                x[self.slices[(k, species, "im")]] = np.imag(data)
        return x

    def _mode_field(
        self,
        coeff: dict[int, np.ndarray],
        species: str,
        k: int,
        where: str = "q",
    ) -> np.ndarray:
        if k == 0:
            basis = self.sb if species == "S" else self.ib[0]
            mat = basis.val if where == "q" else basis.ep
            return mat @ coeff[0]
        sign = 1 if k > 0 else -1
        kk = abs(k)
        basis = self.sb if species == "S" else self.ib[kk]
        mat = basis.val if where == "q" else basis.ep
        field = mat @ coeff[kk]
        return field if sign > 0 else np.conj(field)

    def reconstruct_quadrature(self, S, I) -> tuple[np.ndarray, np.ndarray]:
        Sq = np.tile(np.real(self.sb.val @ S[0]), (self.Q, 1))
        Iq = np.tile(np.real(self.ib[0].val @ I[0]), (self.Q, 1))
        for k in range(1, self.N + 1):
            sf = self.sb.val @ S[k]
            inf = self.ib[k].val @ I[k]
            Sq += 2.0 * np.real(self.exp_pos[k][:, None] * sf[None, :])
            Iq += 2.0 * np.real(self.exp_pos[k][:, None] * inf[None, :])
        return Sq, Iq

    def reaction_modes(self, S, I, Q_override: int | None = None):
        if Q_override is None or Q_override == self.Q:
            Sq, Iq = self.reconstruct_quadrature(S, I)
            s = self.s
        else:
            s = np.arange(Q_override) / Q_override
            Sq = np.tile(np.real(self.sb.val @ S[0]), (Q_override, 1))
            Iq = np.tile(np.real(self.ib[0].val @ I[0]), (Q_override, 1))
            for k in range(1, self.N + 1):
                e = np.exp(1j * 2.0 * np.pi * k * s)
                Sq += 2.0 * np.real(e[:, None] * (self.sb.val @ S[k])[None, :])
                Iq += 2.0 * np.real(e[:, None] * (self.ib[k].val @ I[k])[None, :])
        den = Sq + Iq
        if np.min(den) <= 1.0e-8:
            den = np.maximum(den, 1.0e-8)
        incidence = self.par.beta * Sq * Iq / den
        FS = (
            Sq * (self.par.rho - self.par.mu * Sq)
            - incidence
            + self.par.theta * self.par.gamma * Iq
        )
        FI = incidence - self.par.gamma * Iq
        fsm, fim = {}, {}
        for k in range(self.N + 1):
            e = np.exp(-1j * 2.0 * np.pi * k * s)
            fsm[k] = np.mean(FS * e[:, None], axis=0)
            fim[k] = np.mean(FI * e[:, None], axis=0)
        return fsm, fim, float(np.min(Sq)), float(np.min(Iq)), float(np.min(Sq + Iq))

    def residual(self, x: np.ndarray, Q_override: int | None = None) -> np.ndarray:
        S, I = self.unpack(x)
        fsm, fim, minS, minI, minTot = self.reaction_modes(S, I, Q_override)
        bad = max(0.0, 1.0e-5 - minTot)
        rparts = []
        for k in range(self.N + 1):
            As = self.par.dS * self.mesh.K + self.par.dS * self.par.a * (
                self.mesh.ep.T @ self.mesh.ep
            )
            if k:
                As = As + 1j * 2.0 * np.pi * k / self.eps * self.mesh.M
            rs = As @ S[k] - self.sb.val.conj().T @ (self.mesh.wq * fsm[k])
            ib = self.ib[k]
            Ai = self.par.dI * ib.K + self.par.dI * self.par.bbar * ib.B
            if k:
                Ai = Ai + 1j * 2.0 * np.pi * k / self.eps * ib.M
            ri = Ai @ I[k] - ib.val.conj().T @ (self.mesh.wq * fim[k])
            for ell in range(-self.par.K, self.par.K + 1):
                qell = self.par.q(ell)
                j = k - ell
                if ell == 0 or abs(j) > self.N:
                    continue
                fj = self._mode_field(I, "I", j, "ep")
                ri += self.par.dI * qell * (ib.ep.conj().T @ fj)
            if k == 0:
                rparts.extend([np.real(rs), np.real(ri)])
            else:
                rparts.extend([np.real(rs), np.imag(rs), np.real(ri), np.imag(ri)])
        r = np.concatenate(rparts)
        if bad > 0:
            r = r + (1.0e4 * bad) * np.sign(r + 1.0e-30)
        return r

    def jacobian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        S, I = self.unpack(x)
        dS, dI = self.unpack(v)
        Sq, Iq = self.reconstruct_quadrature(S, I)
        dSq, dIq = self.reconstruct_quadrature(dS, dI)
        den = Sq + Iq
        if np.min(den) <= 1.0e-5:
            raise ValueError(
                "The analytic Jacobian is restricted to the positive branch."
            )
        dincidence = self.par.beta * (Iq * Iq * dSq + Sq * Sq * dIq) / (den * den)
        dFS = (
            (self.par.rho - 2.0 * self.par.mu * Sq) * dSq
            - dincidence
            + self.par.theta * self.par.gamma * dIq
        )
        dFI = dincidence - self.par.gamma * dIq
        dfsm, dfim = {}, {}
        for k in range(self.N + 1):
            phase = np.exp(-1j * 2.0 * np.pi * k * self.s)
            dfsm[k] = np.mean(dFS * phase[:, None], axis=0)
            dfim[k] = np.mean(dFI * phase[:, None], axis=0)

        rparts = []
        boundary_s = self.mesh.ep.T @ self.mesh.ep
        for k in range(self.N + 1):
            As = self.par.dS * self.mesh.K + self.par.dS * self.par.a * boundary_s
            if k:
                As = As + 1j * 2.0 * np.pi * k / self.eps * self.mesh.M
            rs = As @ dS[k] - self.sb.val.conj().T @ (self.mesh.wq * dfsm[k])
            ib = self.ib[k]
            Ai = self.par.dI * ib.K + self.par.dI * self.par.bbar * ib.B
            if k:
                Ai = Ai + 1j * 2.0 * np.pi * k / self.eps * ib.M
            ri = Ai @ dI[k] - ib.val.conj().T @ (self.mesh.wq * dfim[k])
            for ell in range(-self.par.K, self.par.K + 1):
                qell = self.par.q(ell)
                j = k - ell
                if ell == 0 or abs(j) > self.N:
                    continue
                trace = self._mode_field(dI, "I", j, "ep")
                ri += self.par.dI * qell * (ib.ep.conj().T @ trace)
            if k == 0:
                rparts.extend([np.real(rs), np.real(ri)])
            else:
                rparts.extend([np.real(rs), np.imag(rs), np.real(ri), np.imag(ri)])
        return np.concatenate(rparts)

    def linear_block_preconditioner(self) -> LinearOperator:
        boundary_s = self.mesh.ep.T @ self.mesh.ep
        s_lu, i_lu = {}, {}
        for k in range(self.N + 1):
            As = self.par.dS * self.mesh.K + self.par.dS * self.par.a * boundary_s
            if k:
                As = As + 1j * 2.0 * np.pi * k / self.eps * self.mesh.M
            ib = self.ib[k]
            Ai = self.par.dI * ib.K + self.par.dI * self.par.bbar * ib.B
            if k:
                Ai = Ai + 1j * 2.0 * np.pi * k / self.eps * ib.M
            s_lu[k] = splu(csc_matrix(As))
            i_lu[k] = splu(csc_matrix(Ai))

        def apply(rhs: np.ndarray) -> np.ndarray:
            out = np.zeros_like(rhs)
            out[self.slices[(0, "S")]] = np.real(
                s_lu[0].solve(rhs[self.slices[(0, "S")]])
            )
            out[self.slices[(0, "I")]] = np.real(
                i_lu[0].solve(rhs[self.slices[(0, "I")]])
            )
            for k in range(1, self.N + 1):
                for species, factors in (("S", s_lu), ("I", i_lu)):
                    re = self.slices[(k, species, "re")]
                    im = self.slices[(k, species, "im")]
                    solved = factors[k].solve(rhs[re] + 1j * rhs[im])
                    out[re], out[im] = np.real(solved), np.imag(solved)
            return out

        return LinearOperator((self.ndof, self.ndof), matvec=apply, dtype=float)

    def solve_newton_gmres(
        self, x0: np.ndarray, residual_tol: float, max_newton: int = 20
    ) -> tuple[np.ndarray, str | None]:
        x = x0.copy()
        preconditioner = self.linear_block_preconditioner()

        r = self.residual(x)
        res_norm = float(np.linalg.norm(r))
        for it in range(max_newton + 1):
            if res_norm <= residual_tol:
                return x, None
            if it == max_newton:
                break
            jacobian = LinearOperator(
                (self.ndof, self.ndof),
                matvec=lambda direction: self.jacobian_vector_product(x, direction),
                dtype=float,
            )
            forcing = min(0.1, max(1.0e-7, math.sqrt(res_norm)))
            step, lin_info = gmres(
                jacobian,
                -r,
                M=preconditioner,
                rtol=forcing,
                atol=0.0,
                restart=60,
                maxiter=30,
            )
            if lin_info != 0:
                return x, f"GMRES linear solve failed (info={lin_info})"
            alpha = 1.0
            for _ in range(14):
                candidate = x + alpha * step
                res_cand = self.residual(candidate)
                norm_cand = float(np.linalg.norm(res_cand))
                if norm_cand < (1.0 - 1.0e-4 * alpha) * res_norm:
                    x, r, res_norm = candidate, res_cand, norm_cand
                    break
                alpha *= 0.5
            else:
                return x, "Newton line search failed"
        return x, "Newton iteration limit reached"

    def initial(self, steady: np.ndarray, dudb: np.ndarray) -> np.ndarray:
        nn = self.mesh.nn
        S = {0: steady[:nn] - self.par.kappa(self.N) * math.sqrt(self.eps) * dudb[:nn]}
        I = {0: steady[nn:] - self.par.kappa(self.N) * math.sqrt(self.eps) * dudb[nn:]}

        for k in range(1, self.N + 1):
            S[k] = np.zeros(nn, dtype=complex)
            ib = self.ib[k]
            rhs = (
                -self.par.dI * self.par.q(k) * (ib.ep.conj().T @ (self.mesh.ep @ I[0]))
            )
            Ai = (
                self.par.dI * ib.K
                + self.par.dI * self.par.bbar * ib.B
                + 1j * 2.0 * np.pi * k / self.eps * ib.M
            )
            I[k] = splu(csc_matrix(Ai)).solve(np.asarray(rhs).ravel())
        return self.pack(S, I)

    def zero_energy(self, v: np.ndarray) -> float:
        nn = self.mesh.nn
        Es = (
            self.par.dS * self.mesh.K
            + self.par.dS * self.par.a * (self.mesh.ep.T @ self.mesh.ep)
            + self.mesh.M
        )
        Ei = (
            self.par.dI * self.mesh.K
            + self.par.dI * self.par.bbar * (self.mesh.ep.T @ self.mesh.ep)
            + self.mesh.M
        )
        return math.sqrt(
            max(0.0, float(np.real(v[:nn] @ (Es @ v[:nn]) + v[nn:] @ (Ei @ v[nn:]))))
        )


def steady_and_derivative(ne: int, par: Parameters, volume_points: int = 12):
    mesh = P1Mesh(ne, ngauss=volume_points)
    phi = csr_matrix(mesh.phi)
    stiffness = csc_matrix(mesh.K)
    boundary = csc_matrix(mesh.ep.T @ mesh.ep)
    linear_s = par.dS * stiffness + par.dS * par.a * boundary
    linear_i = par.dI * stiffness + par.dI * par.bbar * boundary
    state = np.concatenate([1.2 * np.ones(mesh.nn), 0.8 * np.ones(mesh.nn)])

    def residual_and_jacobian(u: np.ndarray):
        S, I = u[: mesh.nn], u[mesh.nn :]
        Sq, Iq = mesh.phi @ S, mesh.phi @ I
        den = Sq + Iq
        incidence = par.beta * Sq * Iq / den
        FS = Sq * (par.rho - par.mu * Sq) - incidence + par.theta * par.gamma * Iq
        FI = incidence - par.gamma * Iq
        residual = np.concatenate(
            [
                linear_s @ S - mesh.phi.T @ (mesh.wq * FS),
                linear_i @ I - mesh.phi.T @ (mesh.wq * FI),
            ]
        )
        inc_S = par.beta * Iq * Iq / (den * den)
        inc_I = par.beta * Sq * Sq / (den * den)
        derivatives = (
            par.rho - 2.0 * par.mu * Sq - inc_S,
            -inc_I + par.theta * par.gamma,
            inc_S,
            inc_I - par.gamma,
        )
        blocks = [
            phi.T @ diags(mesh.wq * derivative) @ phi for derivative in derivatives
        ]
        jacobian = bmat(
            [
                [linear_s - blocks[0], -blocks[1]],
                [-blocks[2], linear_i - blocks[3]],
            ],
            format="csc",
        )
        return residual, jacobian

    for _ in range(30):
        residual, jacobian = residual_and_jacobian(state)
        res_norm = float(np.linalg.norm(residual))
        if res_norm <= 1.0e-11:
            break
        step = splu(jacobian).solve(-residual)
        alpha = 1.0
        for _ in range(16):
            candidate = state + alpha * step
            if np.min(candidate) > 1.0e-8:
                res_cand, _ = residual_and_jacobian(candidate)
                if np.linalg.norm(res_cand) < (1.0 - 1.0e-4 * alpha) * res_norm:
                    state = candidate
                    break
            alpha *= 0.5
        else:
            raise RuntimeError("Sparse steady Newton line search failed.")
    else:
        raise RuntimeError("Sparse steady Newton did not converge.")

    _, jacobian = residual_and_jacobian(state)
    I = state[mesh.nn :]
    rhs_b = np.concatenate(
        [
            np.zeros(mesh.nn),
            par.dI * (boundary @ I),
        ]
    )
    return state, splu(jacobian).solve(-rhs_b)
