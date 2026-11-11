from __future__ import annotations

import math
import os
import tempfile
from dataclasses import replace
from math import sqrt

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.spatial import Delaunay
from threadpoolctl import threadpool_limits

from common import (
    BUILD,
    REFERENCE,
    TABLES,
    FourierGalerkin,
    Parameters,
    save_table,
    steady_and_derivative,
)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        table1, modal, profile, mechanism = interval()
        save_table(
            table1, REFERENCE / "table1.csv", TABLES / "table1.csv", (None, 4, 4, 4, 4)
        )
        draw_figure(modal, profile, mechanism)
        table2 = disk()
        save_table(
            table2, REFERENCE / "table2.csv", TABLES / "table2.csv", (None, 4, 4, 3)
        )
    print("Tables 1–2 and Figure 1 checked")


def interval():
    par = Parameters()
    steady, du_db = steady_and_derivative(10, par, volume_points=192)
    table, modal, profile, mechanism = [], [], [], []
    periods = (0.04, 0.02, 0.01, 0.005, 0.0025)
    x = np.linspace(0, 1, 201)

    for group, scale in (("low", 0.75), ("base", 1), ("high", 1.20)):
        p = replace(par, q1=scale * par.q1, q2=scale * par.q2)
        for eps in periods if group == "base" else (0.0025,):
            for N in (1, 2, 3):
                model, I, E_mn, E_cor, kappa_hat = solve(
                    eps,
                    N,
                    p,
                    steady,
                    du_db,
                    Q=160,
                )
                modal.append(
                    dict(
                        group=group,
                        epsilon=eps,
                        N=N,
                        kappa_hat=kappa_hat,
                        kappa=p.kappa(),
                    )
                )
                if group == "base" and N == 3:
                    table.append((eps, E_mn, E_mn / sqrt(eps), E_cor, E_cor / eps))
                    if eps in (0.04, 0.01, 0.0025):
                        A_I = np.sqrt(
                            2
                            * sum(
                                abs(model.ib[k].eval(x)[0] @ I[k]) ** 2
                                for k in range(1, N + 1)
                            )
                        )
                        profile.extend(
                            dict(
                                epsilon=eps,
                                x=xx,
                                A_I_norm=y,
                            )
                            for xx, y in zip(x, A_I / A_I.max())
                        )

    for eps in (0.01, 0.0025, 0.000625, 0.00015625, 0.0000390625):
        for method, enriched in (("enriched", True), ("plain", False)):
            _, _, _, _, kappa_hat = solve(
                eps,
                3,
                par,
                steady,
                du_db,
                Q=80,
                enriched=enriched,
            )
            mechanism.append(
                dict(
                    method=method,
                    epsilon=eps,
                    kappa_hat=kappa_hat,
                    kappa=par.kappa(),
                )
            )
    return table, modal, profile, mechanism


def disk(eps_values=(0.000625, 0.00015625, 0.0000390625)):
    par = Parameters()
    mesh = CurvedDiskMesh(14, normal_order=64, tangential_order=4, boundary_order=8)
    assert (mesh.nn, mesh.nt) == (811, 1532)
    steady, du_db = accurate_steady(mesh, par)
    n = mesh.nn
    Es = par.dS * mesh.K + par.dS * par.a * mesh.B + mesh.M
    Ei = par.dI * mesh.K + par.dI * par.bbar * mesh.B + mesh.M
    rows = []
    for eps in eps_values:
        print(f"Disk solve for epsilon={eps:g}", flush=True)
        model = FourierGalerkin2D(mesh, 3, eps, par, phase_points=64)
        coef = model.solve(model.initial(steady, du_db), tolerance=1e-13, max_newton=20)
        S, I = model.unpack(coef)
        delta = np.r_[S[0].real, I[0].real] - steady
        kappa = par.kappa(3)
        E_mn = model.zero_energy(delta)
        E_cor = model.zero_energy(delta + kappa * sqrt(eps) * du_db)
        inner = float(delta[:n] @ (Es @ du_db[:n]) + delta[n:] @ (Ei @ du_db[n:]))
        dh = model.zero_energy(du_db)
        kappa_hat = -inner / (sqrt(eps) * dh * dh)
        rows.append(
            (eps, E_mn / sqrt(eps), E_cor / eps, 100 * abs(kappa_hat - kappa) / kappa)
        )
    return rows


def solve(eps, N, par, steady, du_db, Q, enriched=True):
    model = FourierGalerkin(
        10,
        N,
        eps,
        par,
        enriched=enriched,
        phase_points=Q,
        volume_points=192,
    )
    coef, failure = model.solve_newton_gmres(
        model.initial(steady, du_db),
        residual_tol=1e-13,
        max_newton=90,
    )
    if failure is not None:
        raise RuntimeError(
            f"Interval solve failed: epsilon={eps:g}, N={N}, Q={Q} ({failure})"
        )

    S, I = model.unpack(coef)
    Sq, Iq = model.reconstruct_quadrature(S, I)
    if min(Sq.min(), Iq.min()) <= 0:
        raise RuntimeError(f"Interval solve failed: epsilon={eps:g}, N={N}, Q={Q}")

    delta = np.r_[S[0].real, I[0].real] - steady
    E_mn = model.zero_energy(delta)
    E_cor = model.zero_energy(delta + par.kappa(N) * sqrt(eps) * du_db)
    inner = (
        model.zero_energy(delta + du_db) ** 2 - model.zero_energy(delta - du_db) ** 2
    ) / 4
    kappa_hat = -inner / (sqrt(eps) * model.zero_energy(du_db) ** 2)
    return model, I, E_mn, E_cor, kappa_hat


def disk_beta(points, beta):
    return beta * (1.0 + 0.15 * points[:, 0] + 0.10 * points[:, 1])


def cutoff(distance, width=0.35):
    distance = np.asarray(distance)
    t = distance / width
    active = (distance >= -1e-12) & (distance < width)
    chi = np.where(active, 1 - 3 * t * t + 2 * t * t * t, 0.0)
    dchi = np.where(active, (-6 * t + 6 * t * t) / width, 0.0)
    return chi, dchi


def radial_layer(points, nboundary, k, eps, dI):
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=1)
    distance = 1 - radius
    chi, dchi = cutoff(distance)
    selected = np.flatnonzero(chi != 0.0)
    shape = (len(points), nboundary)
    if not len(selected):
        z = csr_matrix(shape, dtype=complex)
        return z, z.copy(), z.copy()
    xy = points[selected]
    rr = radius[selected]
    angle = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi)
    da = 2 * np.pi / nboundary
    u = angle / da
    left = np.floor(u).astype(int) % nboundary
    local = u - np.floor(u)
    right = (left + 1) % nboundary
    psi = np.column_stack([1 - local, local]).ravel()
    dpsi = np.tile([-1 / da, 1 / da], len(selected))
    mu = np.sqrt(2j * np.pi * k / (dI * eps))
    if mu.real < 0:
        mu = -mu
    exponential = np.exp(-mu * distance[selected])
    radial = exponential * chi[selected]
    rows = np.repeat(selected, 2)
    cols = np.column_stack([left, right]).ravel()
    E = csr_matrix((np.repeat(radial, 2) * psi, (rows, cols)), shape=shape)
    radial_derivative = exponential * (mu * chi[selected] - dchi[selected])
    er = xy / rr[:, None]
    grad_angle = np.column_stack([-xy[:, 1], xy[:, 0]]) / (rr * rr)[:, None]
    dr = np.repeat(radial_derivative, 2) * psi
    dt = np.repeat(radial, 2) * dpsi
    Ex = csr_matrix(
        (
            dr * np.repeat(er[:, 0], 2) + dt * np.repeat(grad_angle[:, 0], 2),
            (rows, cols),
        ),
        shape=shape,
    )
    Ey = csr_matrix(
        (
            dr * np.repeat(er[:, 1], 2) + dt * np.repeat(grad_angle[:, 1], 2),
            (rows, cols),
        ),
        shape=shape,
    )
    return E, Ex, Ey


class CurvedDiskMesh:
    def __init__(
        self, resolution, normal_order=12, tangential_order=4, boundary_order=6
    ):
        self.resolution = int(resolution)
        self.normal_order = int(normal_order)
        self.tangential_order = int(tangential_order)
        self.boundary_order = int(boundary_order)
        self.core_radius = 0.60
        self.h_nominal = 1 / self.resolution
        self.nboundary = max(24, int(math.ceil(2 * np.pi * self.resolution)))
        self.da = 2 * np.pi / self.nboundary
        angle = np.arange(self.nboundary) * self.da
        unit = np.column_stack([np.cos(angle), np.sin(angle)])
        grid = np.arange(
            -self.core_radius + self.h_nominal, self.core_radius, self.h_nominal
        )
        xx, yy = np.meshgrid(grid, grid, indexing="xy")
        interior = np.column_stack([xx.ravel(), yy.ravel()])
        interior = interior[
            np.linalg.norm(interior, axis=1) < self.core_radius - 0.58 * self.h_nominal
        ]
        self.core_nodes = np.vstack([interior, self.core_radius * unit])
        self.core_nn = len(self.core_nodes)
        self.core_boundary = np.arange(len(interior), self.core_nn)
        self.core_triangulation = Delaunay(self.core_nodes)
        self.core_triangles = self.core_triangulation.simplices.copy()
        nr = max(2, int(math.ceil((1 - self.core_radius) * self.resolution)))
        self.radii = np.linspace(self.core_radius, 1, nr + 1)
        self.ring_nodes = [self.core_boundary]
        all_nodes = [self.core_nodes]
        next_node = self.core_nn
        for radius in self.radii[1:]:
            all_nodes.append(radius * unit)
            self.ring_nodes.append(np.arange(next_node, next_node + self.nboundary))
            next_node += self.nboundary
        self.nodes = np.vstack(all_nodes)
        self.nn = len(self.nodes)
        self.boundary_nodes = self.ring_nodes[-1]
        self.curved_core = {}
        self.sector_core = {}
        for it, tri in enumerate(self.core_triangles):
            outside = [int(i) for i in tri if i >= len(interior)]
            if len(outside) == 2:
                ia, ib = [i - len(interior) for i in outside]
                j = ia if (ia + 1) % self.nboundary == ib else ib
                assert (j + 1) % self.nboundary in (ia, ib)
                apex = next(int(i) for i in tri if i not in outside)
                self.curved_core[it] = (
                    int(self.core_boundary[j]),
                    int(self.core_boundary[(j + 1) % self.nboundary]),
                    apex,
                    j,
                )
                self.sector_core[j] = it
        assert len(self.curved_core) == self.nboundary
        self.annular_triangles = []
        for ell in range(nr):
            for j in range(self.nboundary):
                jj = (j + 1) % self.nboundary
                lo, hi = self.ring_nodes[ell], self.ring_nodes[ell + 1]
                self.annular_triangles.extend(
                    [(lo[j], hi[j], hi[jj]), (lo[j], hi[jj], lo[jj])]
                )
        self.triangles = np.vstack(
            [self.core_triangles, np.array(self.annular_triangles)]
        )
        self.nt = len(self.triangles)
        self._assemble_volume()
        self._assemble_boundary()

    def core_map(self, it, s, t):
        s, t = np.broadcast_arrays(
            np.asarray(s, dtype=float), np.asarray(t, dtype=float)
        )
        i0, i1, ic, j = self.curved_core[it]
        a, b, c = self.nodes[[i0, i1, ic]]
        angle = (j + s) * self.da
        arc = self.core_radius * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        arc_s = (
            self.core_radius
            * self.da
            * np.stack([-np.sin(angle), np.cos(angle)], axis=-1)
        )
        chord = (1 - s[..., None]) * a + s[..., None] * b
        delta = arc - chord
        delta_s = arc_s - (b - a)
        w = 1 - t
        points = w[..., None] * chord + t[..., None] * c + w[..., None] ** 2 * delta
        Fs = w[..., None] * (b - a) + w[..., None] ** 2 * delta_s
        Ft = c - chord - 2 * w[..., None] * delta
        return points, Fs, Ft

    def _assemble_volume(self):
        points, weights, columns, values, gxs, gys = [], [], [], [], [], []

        def append(x, w, ids, val, gx, gy):
            points.append(x)
            weights.append(w)
            columns.append(np.broadcast_to(np.asarray(ids), (len(w), 3)).copy())
            values.append(val)
            gxs.append(gx)
            gys.append(gy)

        z, w = leggauss(max(5, self.tangential_order))
        u, wu = (z + 1) / 2, w / 2
        tt, ss = np.meshgrid(u, u, indexing="ij")
        wt, ws = np.meshgrid(wu, wu, indexing="ij")
        t, s = tt.ravel(), ss.ravel()
        qw = (wt * ws).ravel()
        val = np.column_stack([(1 - t) * (1 - s), (1 - t) * s, t])
        ds = np.column_stack([-(1 - t), 1 - t, np.zeros(len(s))])
        dt = np.column_stack([-(1 - s), -s, np.ones(len(s))])
        minimum_jacobian = math.inf
        for it, tri in enumerate(self.core_triangles):
            if it in self.curved_core:
                ids = self.curved_core[it][:3]
                x, Fs, Ft = self.core_map(it, s, t)
                jac = Fs[:, 0] * Ft[:, 1] - Fs[:, 1] * Ft[:, 0]
                assert np.min(jac) > 0
                minimum_jacobian = min(minimum_jacobian, float(np.min(jac / (1 - t))))
                gx = (ds * Ft[:, 1, None] - dt * Fs[:, 1, None]) / jac[:, None]
                gy = (-ds * Ft[:, 0, None] + dt * Fs[:, 0, None]) / jac[:, None]
            else:
                ids = tri
                p = self.nodes[tri]
                x = val @ p
                Fs = (1 - t)[:, None] * (p[1] - p[0])
                Ft = p[2] - ((1 - s)[:, None] * p[0] + s[:, None] * p[1])
                jac = Fs[:, 0] * Ft[:, 1] - Fs[:, 1] * Ft[:, 0]
                gx = (ds * Ft[:, 1, None] - dt * Fs[:, 1, None]) / jac[:, None]
                gy = (-ds * Ft[:, 0, None] + dt * Fs[:, 0, None]) / jac[:, None]
            append(x, qw * np.abs(jac), ids, val, gx, gy)
        zn, wn = leggauss(self.normal_order)
        za, wa = leggauss(self.tangential_order)
        rho, eta_base = np.meshgrid((zn + 1) / 2, (za + 1) / 2, indexing="ij")
        radial_w, angle_w = np.meshgrid(wn / 2, wa / 2, indexing="ij")
        rho, eta_base = rho.ravel(), eta_base.ravel()
        base_w = (radial_w * angle_w).ravel()
        for ell, (r0, r1) in enumerate(zip(self.radii[:-1], self.radii[1:])):
            dr = r1 - r0
            radius = r0 + dr * rho
            for j in range(self.nboundary):
                jj = (j + 1) % self.nboundary
                lo, hi = self.ring_nodes[ell], self.ring_nodes[ell + 1]
                for kind in (0, 1):
                    if kind == 0:
                        eta = rho * eta_base
                        factor = rho
                        ids = [lo[j], hi[j], hi[jj]]
                        shape = np.column_stack([1 - rho, rho - eta, eta])
                        gr = np.array([-1, 1, 0]) / dr
                        ga = np.array([0, -1, 1]) / self.da
                    else:
                        eta = rho + (1 - rho) * eta_base
                        factor = 1 - rho
                        ids = [lo[j], hi[jj], lo[jj]]
                        shape = np.column_stack([1 - eta, rho, eta - rho])
                        gr = np.array([0, 1, -1]) / dr
                        ga = np.array([-1, 0, 1]) / self.da
                    angle = (j + eta) * self.da
                    cs, sn = np.cos(angle), np.sin(angle)
                    x = radius[:, None] * np.column_stack([cs, sn])
                    gx = cs[:, None] * gr - sn[:, None] / radius[:, None] * ga
                    gy = sn[:, None] * gr + cs[:, None] / radius[:, None] * ga
                    append(
                        x, base_w * dr * self.da * factor * radius, ids, shape, gx, gy
                    )
        self.xq = np.vstack(points)
        self.wq = np.concatenate(weights)
        cols = np.vstack(columns).ravel()
        rows = np.repeat(np.arange(len(self.xq)), 3)
        shape = (len(self.xq), self.nn)
        self.phi = csr_matrix((np.vstack(values).ravel(), (rows, cols)), shape=shape)
        self.gx = csr_matrix((np.vstack(gxs).ravel(), (rows, cols)), shape=shape)
        self.gy = csr_matrix((np.vstack(gys).ravel(), (rows, cols)), shape=shape)
        W = diags(self.wq, format="csc")
        self.M = csc_matrix(self.phi.T @ W @ self.phi)
        self.K = csc_matrix(self.gx.T @ W @ self.gx + self.gy.T @ W @ self.gy)
        self.H = self.M + self.K
        self.minimum_core_map_jacobian = minimum_jacobian

    def _assemble_boundary(self):
        z, w = leggauss(self.boundary_order)
        local = (z + 1) / 2
        angles = (np.arange(self.nboundary)[:, None] + local) * self.da
        self.bq = np.column_stack([np.cos(angles.ravel()), np.sin(angles.ravel())])
        self.bw = np.tile(w / 2 * self.da, self.nboundary)
        rows = np.repeat(np.arange(len(self.bq)), 2)
        bn = self.boundary_nodes
        ids = np.column_stack([bn, np.roll(bn, -1)])
        cols = np.repeat(ids, self.boundary_order, axis=0).ravel()
        val = np.tile(np.column_stack([1 - local, local]), (self.nboundary, 1)).ravel()
        self.bphi = csr_matrix((val, (rows, cols)), shape=(len(self.bq), self.nn))
        self.B = csc_matrix(self.bphi.T @ diags(self.bw) @ self.bphi)


class StandardBasis:
    def __init__(self, mesh: CurvedDiskMesh):
        self.mesh = mesh
        self.n = mesh.nn
        self.M, self.K, self.B = (
            mesh.M.astype(complex),
            mesh.K.astype(complex),
            mesh.B.astype(complex),
        )

    def volume_field(self, coef: np.ndarray) -> np.ndarray:
        return self.mesh.phi @ coef

    def boundary_field(self, coef: np.ndarray) -> np.ndarray:
        return self.mesh.bphi @ coef

    def volume_project(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.mesh.phi.T @ (self.mesh.wq * values)).ravel()

    def boundary_project(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.mesh.bphi.T @ (self.mesh.bw * values)).ravel()


class EnrichedBasis:
    def __init__(self, mesh: CurvedDiskMesh, k: int, eps: float, dI: float):
        self.mesh, self.k, self.eps, self.dI = mesh, k, eps, dI
        E, Ex, Ey = radial_layer(mesh.xq, mesh.nboundary, k, eps, dI)
        X = mesh.bphi[:, mesh.boundary_nodes].astype(complex)
        W = diags(mesh.wq, format="csc")
        Wb = diags(mesh.bw, format="csc")
        CM = csc_matrix(mesh.phi.T @ W @ E)
        CK = csc_matrix(mesh.gx.T @ W @ Ex + mesh.gy.T @ W @ Ey)
        CB = csc_matrix(mesh.bphi.T @ Wb @ X)
        ME = np.asarray((E.conj().T @ W @ E).toarray())
        KE = np.asarray((Ex.conj().T @ W @ Ex + Ey.conj().T @ W @ Ey).toarray())
        BE = np.asarray((X.conj().T @ Wb @ X).toarray())
        rhs = (CM + CK).toarray()
        P = splu(mesh.H.astype(complex)).solve(rhs)
        gram = ME + KE - rhs.conj().T @ P
        gram = 0.5 * (gram + gram.conj().T)
        eigval, eigvec = np.linalg.eigh(gram)
        threshold = max(1.0e-12, 1.0e-10 * float(np.max(eigval)))
        keep = eigval > threshold
        if not np.any(keep):
            raise RuntimeError("All enrichment directions were eliminated.")
        T = eigvec[:, keep] / np.sqrt(eigval[keep])[None, :]
        self.rank = T.shape[1]
        self.n = mesh.nn + self.rank
        self.P, self.T, self.E, self.X = P, T, E, X

        M, K, B = mesh.M.astype(complex), mesh.K.astype(complex), mesh.B.astype(complex)
        CMd, CKd, CBd = CM.toarray(), CK.toarray(), CB.toarray()
        MP, KP, BP = M @ P, K @ P, B @ P
        Mfz = (CMd - MP) @ T
        Kfz = (CKd - KP) @ T
        Bfz = (CBd - BP) @ T
        Mzz0 = ME - CMd.conj().T @ P - P.conj().T @ CMd + P.conj().T @ MP
        Kzz0 = KE - CKd.conj().T @ P - P.conj().T @ CKd + P.conj().T @ KP
        Bzz0 = BE - CBd.conj().T @ P - P.conj().T @ CBd + P.conj().T @ BP
        Mzz = T.conj().T @ Mzz0 @ T
        Kzz = T.conj().T @ Kzz0 @ T
        Bzz = T.conj().T @ Bzz0 @ T
        self.M = bmat(
            [[M, csc_matrix(Mfz)], [csc_matrix(Mfz.conj().T), csc_matrix(Mzz)]],
            format="csc",
        )
        self.K = bmat(
            [[K, csc_matrix(Kfz)], [csc_matrix(Kfz.conj().T), csc_matrix(Kzz)]],
            format="csc",
        )
        self.B = bmat(
            [[B, csc_matrix(Bfz)], [csc_matrix(Bfz.conj().T), csc_matrix(Bzz)]],
            format="csc",
        )

    def _split(self, coef: np.ndarray):
        alpha = self.T @ coef[self.mesh.nn :]
        effective_fe = coef[: self.mesh.nn] - self.P @ alpha
        return effective_fe, alpha

    def volume_field(self, coef: np.ndarray) -> np.ndarray:
        effective, alpha = self._split(coef)
        return self.mesh.phi @ effective + self.E @ alpha

    def boundary_field(self, coef: np.ndarray) -> np.ndarray:
        effective, alpha = self._split(coef)
        return self.mesh.bphi @ effective + self.X @ alpha

    def volume_project(self, values: np.ndarray) -> np.ndarray:
        standard = np.asarray(self.mesh.phi.T @ (self.mesh.wq * values)).ravel()
        raw = np.asarray(self.E.conj().T @ (self.mesh.wq * values)).ravel()
        enriched = self.T.conj().T @ (raw - self.P.conj().T @ standard)
        return np.concatenate([standard, enriched])

    def boundary_project(self, values: np.ndarray) -> np.ndarray:
        standard = np.asarray(self.mesh.bphi.T @ (self.mesh.bw * values)).ravel()
        raw = np.asarray(self.X.conj().T @ (self.mesh.bw * values)).ravel()
        enriched = self.T.conj().T @ (raw - self.P.conj().T @ standard)
        return np.concatenate([standard, enriched])


class FourierGalerkin2D:
    def __init__(self, mesh, N, eps, par, phase_points=64):
        self.N, self.eps, self.par = N, eps, par
        self.mesh = mesh
        self.sb = StandardBasis(self.mesh)
        self.ib = {0: StandardBasis(self.mesh)}
        for k in range(1, N + 1):
            self.ib[k] = EnrichedBasis(self.mesh, k, eps, par.dI)
        self.Q = phase_points
        self.s = np.arange(self.Q) / self.Q
        self.slices = {}
        current = 0
        self.slices[(0, "S")] = slice(current, current + self.mesh.nn)
        current += self.mesh.nn
        self.slices[(0, "I")] = slice(current, current + self.mesh.nn)
        current += self.mesh.nn
        for k in range(1, N + 1):
            for species, size in (("S", self.mesh.nn), ("I", self.ib[k].n)):
                self.slices[(k, species, "re")] = slice(current, current + size)
                current += size
                self.slices[(k, species, "im")] = slice(current, current + size)
                current += size
        self.ndof = current
        self.beta_q = disk_beta(self.mesh.xq, par.beta)

    def unpack(self, x):
        S = {0: x[self.slices[(0, "S")]].astype(complex)}
        I = {0: x[self.slices[(0, "I")]].astype(complex)}
        for k in range(1, self.N + 1):
            S[k] = x[self.slices[(k, "S", "re")]] + 1j * x[self.slices[(k, "S", "im")]]
            I[k] = x[self.slices[(k, "I", "re")]] + 1j * x[self.slices[(k, "I", "im")]]
        return S, I

    def pack(self, S, I):
        x = np.zeros(self.ndof)
        x[self.slices[(0, "S")]] = np.real(S[0])
        x[self.slices[(0, "I")]] = np.real(I[0])
        for k in range(1, self.N + 1):
            for species, values in (("S", S[k]), ("I", I[k])):
                x[self.slices[(k, species, "re")]] = np.real(values)
                x[self.slices[(k, species, "im")]] = np.imag(values)
        return x

    def _basis(self, species, k):
        return self.sb if species == "S" else self.ib[k]

    def _mode_field(self, coeff, species, k, boundary=False):
        sign, kk = (1, k) if k >= 0 else (-1, -k)
        basis = self._basis(species, kk)
        value = (
            basis.boundary_field(coeff[kk])
            if boundary
            else basis.volume_field(coeff[kk])
        )
        return value if sign > 0 else np.conj(value)

    def reaction_modes(self, S, I, dS=None, dI=None):
        fsm = {k: np.zeros(len(self.mesh.xq), dtype=complex) for k in range(self.N + 1)}
        fim = {k: np.zeros(len(self.mesh.xq), dtype=complex) for k in range(self.N + 1)}
        min_s, min_i, min_total = math.inf, math.inf, math.inf
        Sfields = {k: self.sb.volume_field(S[k]) for k in range(self.N + 1)}
        Ifields = {k: self.ib[k].volume_field(I[k]) for k in range(self.N + 1)}
        if dS is not None:
            dSfields = {k: self.sb.volume_field(dS[k]) for k in range(self.N + 1)}
            dIfields = {k: self.ib[k].volume_field(dI[k]) for k in range(self.N + 1)}
        for s in self.s:
            sq = np.real(Sfields[0]).copy()
            iq = np.real(Ifields[0]).copy()
            if dS is not None:
                dsq = np.real(dSfields[0]).copy()
                diq = np.real(dIfields[0]).copy()
            for k in range(1, self.N + 1):
                phase = np.exp(2j * math.pi * k * s)
                sq += 2.0 * np.real(phase * Sfields[k])
                iq += 2.0 * np.real(phase * Ifields[k])
                if dS is not None:
                    dsq += 2.0 * np.real(phase * dSfields[k])
                    diq += 2.0 * np.real(phase * dIfields[k])
            den = sq + iq
            if np.min(den) <= 1.0e-7:
                raise ValueError("Reconstruction left the positive branch.")
            min_s, min_i, min_total = (
                min(min_s, float(np.min(sq))),
                min(min_i, float(np.min(iq))),
                min(min_total, float(np.min(den))),
            )
            if dS is None:
                incidence = self.beta_q * sq * iq / den
                fs = (
                    sq * (self.par.rho - self.par.mu * sq)
                    - incidence
                    + self.par.theta * self.par.gamma * iq
                )
                fi = incidence - self.par.gamma * iq
            else:
                dinc = self.beta_q * (iq * iq * dsq + sq * sq * diq) / (den * den)
                fs = (
                    (self.par.rho - 2.0 * self.par.mu * sq) * dsq
                    - dinc
                    + self.par.theta * self.par.gamma * diq
                )
                fi = dinc - self.par.gamma * diq
            for k in range(self.N + 1):
                phase = np.exp(-2j * math.pi * k * s) / self.Q
                fsm[k] += phase * fs
                fim[k] += phase * fi
        return fsm, fim, min_s, min_i, min_total

    def residual(self, x):
        S, I = self.unpack(x)
        fsm, fim, _, _, _ = self.reaction_modes(S, I)
        parts = []
        for k in range(self.N + 1):
            omega = 2.0 * math.pi * k / self.eps
            As = self.par.dS * self.sb.K + self.par.dS * self.par.a * self.sb.B
            if k:
                As = As + 1j * omega * self.sb.M
            rs = As @ S[k] - self.sb.volume_project(fsm[k])
            ib = self.ib[k]
            Ai = self.par.dI * ib.K + self.par.dI * self.par.bbar * ib.B
            if k:
                Ai = Ai + 1j * omega * ib.M
            ri = Ai @ I[k] - ib.volume_project(fim[k])
            for ell in range(-self.par.K, self.par.K + 1):
                if ell == 0:
                    continue
                j = k - ell
                if abs(j) <= self.N:
                    ri += (
                        self.par.dI
                        * self.par.q(ell)
                        * ib.boundary_project(
                            self._mode_field(I, "I", j, boundary=True)
                        )
                    )
            if k == 0:
                parts.extend([np.real(rs), np.real(ri)])
            else:
                parts.extend([np.real(rs), np.imag(rs), np.real(ri), np.imag(ri)])
        return np.concatenate(parts)

    def jvp(self, x, v):
        S, I = self.unpack(x)
        dS, dI = self.unpack(v)
        dfsm, dfim, _, _, _ = self.reaction_modes(S, I, dS, dI)
        parts = []
        for k in range(self.N + 1):
            omega = 2.0 * math.pi * k / self.eps
            As = self.par.dS * self.sb.K + self.par.dS * self.par.a * self.sb.B
            if k:
                As = As + 1j * omega * self.sb.M
            rs = As @ dS[k] - self.sb.volume_project(dfsm[k])
            ib = self.ib[k]
            Ai = self.par.dI * ib.K + self.par.dI * self.par.bbar * ib.B
            if k:
                Ai = Ai + 1j * omega * ib.M
            ri = Ai @ dI[k] - ib.volume_project(dfim[k])
            for ell in range(-self.par.K, self.par.K + 1):
                if ell == 0:
                    continue
                j = k - ell
                if abs(j) <= self.N:
                    ri += (
                        self.par.dI
                        * self.par.q(ell)
                        * ib.boundary_project(
                            self._mode_field(dI, "I", j, boundary=True)
                        )
                    )
            if k == 0:
                parts.extend([np.real(rs), np.real(ri)])
            else:
                parts.extend([np.real(rs), np.imag(rs), np.real(ri), np.imag(ri)])
        return np.concatenate(parts)

    def preconditioner(self):
        s_lu, i_lu = {}, {}
        for k in range(self.N + 1):
            omega = 2.0 * math.pi * k / self.eps
            As = self.par.dS * self.sb.K + self.par.dS * self.par.a * self.sb.B
            Ai = self.par.dI * self.ib[k].K + self.par.dI * self.par.bbar * self.ib[k].B
            if k:
                As = As + 1j * omega * self.sb.M
                Ai = Ai + 1j * omega * self.ib[k].M
            s_lu[k], i_lu[k] = splu(csc_matrix(As)), splu(csc_matrix(Ai))

        def apply(rhs):
            out = np.zeros_like(rhs)
            out[self.slices[(0, "S")]] = np.real(
                s_lu[0].solve(rhs[self.slices[(0, "S")]])
            )
            out[self.slices[(0, "I")]] = np.real(
                i_lu[0].solve(rhs[self.slices[(0, "I")]])
            )
            for k in range(1, self.N + 1):
                for species, factors in (("S", s_lu), ("I", i_lu)):
                    re, im = (
                        self.slices[(k, species, "re")],
                        self.slices[(k, species, "im")],
                    )
                    solved = factors[k].solve(rhs[re] + 1j * rhs[im])
                    out[re], out[im] = np.real(solved), np.imag(solved)
            return out

        return LinearOperator((self.ndof, self.ndof), matvec=apply, dtype=float)

    def initial(self, steady, derivative):
        n = self.mesh.nn
        S = {
            0: steady[:n]
            - self.par.kappa(self.N) * math.sqrt(self.eps) * derivative[:n]
        }
        I = {
            0: steady[n:]
            - self.par.kappa(self.N) * math.sqrt(self.eps) * derivative[n:]
        }
        for k in range(1, self.N + 1):
            S[k] = np.zeros(n, dtype=complex)
            ib = self.ib[k]
            Ai = (
                self.par.dI * ib.K
                + self.par.dI * self.par.bbar * ib.B
                + 1j * 2.0 * math.pi * k / self.eps * ib.M
            )
            trace0 = self.ib[0].boundary_field(I[0])
            rhs = -self.par.dI * self.par.q(k) * ib.boundary_project(trace0)
            I[k] = splu(csc_matrix(Ai)).solve(rhs)
        return self.pack(S, I)

    def solve(self, initial, tolerance=2.0e-8, max_newton=16):
        x = initial.copy()
        preconditioner = self.preconditioner()
        for iteration in range(max_newton + 1):
            residual = self.residual(x)
            norm = float(np.linalg.norm(residual))
            if norm <= tolerance:
                break
            jacobian = LinearOperator(
                (self.ndof, self.ndof), matvec=lambda v: self.jvp(x, v), dtype=float
            )
            step, status = gmres(
                jacobian,
                -residual,
                M=preconditioner,
                rtol=min(0.08, max(2.0e-6, math.sqrt(norm))),
                atol=0.0,
                restart=70,
                maxiter=35,
            )
            if status != 0:
                raise RuntimeError(f"GMRES failed with status {status}.")
            alpha = 1.0
            for _ in range(13):
                candidate = x + alpha * step
                try:
                    candidate_norm = np.linalg.norm(self.residual(candidate))
                except ValueError:
                    candidate_norm = math.inf
                if candidate_norm < (1.0 - 1.0e-4 * alpha) * norm:
                    x = candidate
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Fourier Newton line search failed.")
        residual = self.residual(x)
        S, I = self.unpack(x)
        _, _, min_s, min_i, min_total = self.reaction_modes(S, I)
        if np.linalg.norm(residual) > tolerance or min(min_s, min_i) <= 0:
            raise RuntimeError(f"Disk solve failed for epsilon={self.eps:g}")
        return x

    def zero_energy(self, vector):
        n = self.mesh.nn
        Es = (
            self.par.dS * self.mesh.K
            + self.par.dS * self.par.a * self.mesh.B
            + self.mesh.M
        )
        Ei = (
            self.par.dI * self.mesh.K
            + self.par.dI * self.par.bbar * self.mesh.B
            + self.mesh.M
        )
        value = vector[:n] @ (Es @ vector[:n]) + vector[n:] @ (Ei @ vector[n:])
        return math.sqrt(max(0.0, float(np.real(value))))


def steady_residual_jacobian(mesh, par, u):
    phi = mesh.phi
    S, I = u[: mesh.nn], u[mesh.nn :]
    sq, iq = phi @ S, phi @ I
    den = sq + iq
    beta = disk_beta(mesh.xq, par.beta)
    inc = beta * sq * iq / den
    fs = sq * (par.rho - par.mu * sq) - inc + par.theta * par.gamma * iq
    fi = inc - par.gamma * iq
    ls = par.dS * mesh.K + par.dS * par.a * mesh.B
    li = par.dI * mesh.K + par.dI * par.bbar * mesh.B
    residual = np.r_[ls @ S - phi.T @ (mesh.wq * fs), li @ I - phi.T @ (mesh.wq * fi)]
    incs = beta * iq * iq / (den * den)
    inci = beta * sq * sq / (den * den)
    derivatives = [
        par.rho - 2 * par.mu * sq - incs,
        -inci + par.theta * par.gamma,
        incs,
        inci - par.gamma,
    ]
    blocks = [phi.T @ diags(mesh.wq * d) @ phi for d in derivatives]
    jac = bmat(
        [[ls - blocks[0], -blocks[1]], [-blocks[2], li - blocks[3]]], format="csc"
    )
    return residual, jac


def accurate_steady(mesh, par):
    u = np.r_[np.full(mesh.nn, 0.9), np.full(mesh.nn, 1.8)]
    for it in range(30):
        residual, jac = steady_residual_jacobian(mesh, par, u)
        norm = float(np.linalg.norm(residual))
        if norm < 1e-13:
            break
        step = splu(jac).solve(-residual)
        alpha = 1.0
        for _ in range(20):
            trial = u + alpha * step
            if (
                min(trial) > 0
                and np.linalg.norm(steady_residual_jacobian(mesh, par, trial)[0]) < norm
            ):
                u = trial
                break
            alpha /= 2
        else:
            raise RuntimeError("steady line search failed")
    residual, jac = steady_residual_jacobian(mesh, par, u)
    assert np.linalg.norm(residual) < 2e-13
    lu = splu(jac)
    rhs = np.r_[np.zeros(mesh.nn), -par.dI * (mesh.B @ u[mesh.nn :])]
    derivative = lu.solve(rhs)
    return u, derivative


def draw_figure(modal, profile, mechanism):
    out = BUILD / "figures"
    cache = BUILD / "cache"
    out.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    tempfile.tempdir = str(cache)

    import matplotlib as mpl

    mpl.use("pgf")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    INK = "#30343b"
    COLORS = ("#176b91", "#ce6a22", "#16856b", "#845797", "#a63d52")
    MARKERS = ("o", "s", "^", "D", "v")
    LINES = ("-", "--", "-.", (0, (4, 1, 1, 1)))
    W = 187.19833 / 72.27

    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath,amssymb,bm,upgreek}",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.55,
            "lines.linewidth": 0.80,
            "lines.markersize": 3.4,
            "lines.markeredgewidth": 0.75,
            "lines.markerfacecolor": "white",
            "axes.grid": False,
            "axes.unicode_minus": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
        }
    )

    def panel():
        fig = plt.figure(figsize=(W, W))
        ax = fig.add_axes((0.18, 0.17, 0.80, 0.80))
        ax.set_box_aspect(1)
        ax.tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
            length=2.5,
            width=0.55,
            pad=2,
        )
        ax.tick_params(which="minor", length=1.5, width=0.4)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        return fig, ax

    def legend(ax, **kwargs):
        options = dict(
            frameon=True,
            facecolor="white",
            framealpha=0.80,
            edgecolor="#c0c4c8",
            handlelength=1.45,
            labelspacing=0.22,
            borderpad=0.28,
            handletextpad=0.45,
        )
        options.update(kwargs)
        leg = ax.legend(**options)
        leg.get_frame().set_linewidth(0.42)
        return leg

    def line(ax, x, y, index, label, **kwargs):
        ax.plot(
            x,
            y,
            color=COLORS[index],
            marker=MARKERS[index],
            linestyle=LINES[index % len(LINES)],
            label=label,
            **kwargs,
        )

    def math_number(value, pos=None):
        if value == 0:
            return "$0$"
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / 10**exponent
        if math.isclose(mantissa, 1):
            return rf"$10^{{{exponent}}}$"
        return rf"${mantissa:g}\times10^{{{exponent}}}$"

    def save(fig, name):
        main = fig.axes[0]
        box = main.get_position()
        main.xaxis.label.set_verticalalignment("baseline")
        main.xaxis.set_label_coords(
            box.x0 + box.width / 2,
            0.065,
            transform=fig.transFigure,
        )

        renderer = fig.canvas.get_renderer()
        fig.draw_without_rendering()
        for ax in [main, *main.child_axes]:
            tight = ax.get_tightbbox(renderer)
            bounds = fig.bbox
            if (
                tight.x0 < -1
                or tight.y0 < -1
                or tight.x1 > bounds.x1 + 1
                or tight.y1 > bounds.y1 + 1
            ):
                raise RuntimeError(f"{name}: clipped axes or labels")

            leg = ax.get_legend()
            if leg is not None:
                legend_box = leg.get_window_extent(renderer)
                for plotted in ax.lines:
                    path = plotted.get_path().transformed(plotted.get_transform())
                    if path.intersects_bbox(legend_box, filled=False):
                        raise RuntimeError(
                            f"{name}: legend overlaps {plotted.get_label()}"
                        )
        fig.savefig(out / name, metadata={"Creator": "Matplotlib PGF with pdfLaTeX"})
        plt.close(fig)

    selected = [r for r in modal if r["group"] == "base"]
    fig, ax = panel()
    series = {}
    for i, n in enumerate((1, 2, 3)):
        z = sorted(
            (r for r in selected if int(r["N"]) == n),
            key=lambda r: float(r["epsilon"]),
        )
        x = [float(r["epsilon"]) for r in z]
        y = [float(r["kappa_hat"]) / float(r["kappa"]) for r in z]
        series[n] = x, y
        line(ax, x, y, i, rf"$N={n}$")

    ax.set_xscale("log")
    ax.set(
        xlim=(0.0018, 0.055),
        ylim=(0.74, 1.045),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\widehat\kappa_N/\kappa$",
    )
    ax.axhline(1, color=INK, ls=":", lw=0.7, zorder=0)
    ax.xaxis.set_major_locator(FixedLocator([0.0025, 0.01, 0.04]))
    ax.xaxis.set_major_formatter(FuncFormatter(math_number))
    legend(
        ax,
        loc="center",
        bbox_to_anchor=(0.58, 0.765),
        ncol=3,
        columnspacing=0.45,
        handlelength=1.1,
    )

    inset = ax.inset_axes([0.15, 0.29, 0.72, 0.39])
    for i, n in enumerate((2, 3), 1):
        x, y = series[n]
        line(inset, x, y, i, None, ms=2.7, lw=0.65)
    inset.axhline(1, color=INK, ls=":", lw=0.55)
    inset.set_xscale("log")
    inset.set(xlim=(0.0018, 0.055), ylim=(0.9955, 1.0008), yticks=[0.996, 1.0])
    inset.tick_params(which="both", direction="in", labelsize=5.8, pad=1, length=1.5)
    inset.xaxis.set_minor_formatter(NullFormatter())
    inset.xaxis.set_major_locator(FixedLocator([0.0025, 0.04]))
    inset.xaxis.set_major_formatter(FuncFormatter(math_number))
    save(fig, "fig1a.pdf")

    fig, ax = panel()
    ax.axvspan(2, 3.3, color="#edf3f2", zorder=0)
    for i, (group, name) in enumerate(
        (
            ("low", r"$0.75q$"),
            ("base", r"$q$"),
            ("high", r"$1.20q$"),
        )
    ):
        z = sorted(
            (r for r in modal if r["group"] == group and float(r["epsilon"]) == 0.0025),
            key=lambda r: int(r["N"]),
        )
        x = np.array([int(r["N"]) for r in z])
        y = [abs(float(r["kappa_hat"]) / float(r["kappa"]) - 1) for r in z]
        line(ax, x, y, i, name)

    ax.set_yscale("log")
    ax.set(
        xlim=(0.7, 3.3),
        ylim=(8e-6, 0.55),
        xticks=[1, 2, 3],
        xlabel=r"$N$",
        ylabel=r"$|\kappa-\widehat\kappa_N|/\kappa$",
    )
    ax.text(2.45, 0.25, r"$N\ge K$", ha="center", va="top", fontsize=7, color="#59756d")
    legend(ax, loc="lower left")
    save(fig, "fig1b.pdf")

    fig, ax = panel()
    for i, eps in enumerate((0.04, 0.01, 0.0025)):
        z = sorted(
            (r for r in profile if float(r["epsilon"]) == eps),
            key=lambda r: float(r["x"]),
        )
        x = np.array([float(r["x"]) for r in z])
        y = np.array([float(r["A_I_norm"]) for r in z])
        line(ax, x, y, i, rf"$\varepsilon={eps:g}$", markevery=20)

    ax.set(
        xlim=(-0.015, 1.015),
        ylim=(-0.025, 1.07),
        xticks=[0, 0.5, 1],
        xlabel=r"$x$",
        ylabel=r"$A_{\mathrm I}(x)/\max_x A_{\mathrm I}(x)$",
    )
    legend(ax, loc="upper center")
    save(fig, "fig1c.pdf")

    fig, ax = panel()
    for i, method in enumerate(("enriched", "plain")):
        z = sorted(
            (r for r in mechanism if r["method"] == method),
            key=lambda r: float(r["epsilon"]),
        )
        x = np.array([float(r["epsilon"]) for r in z])
        y = np.array([float(r["kappa_hat"]) / float(r["kappa"]) for r in z])
        ax.plot(
            x,
            y,
            c=COLORS[i],
            marker=("^", "D")[i],
            ls=LINES[i],
            label=method.capitalize(),
        )

    ax.axhline(1, color=INK, ls=":", lw=0.65)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlim=(2.8e-5, 0.015),
        ylim=(2e-6, 1.8),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\widehat\kappa_N/\kappa$",
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    legend(ax, loc="lower right")
    save(fig, "fig1d.pdf")


if __name__ == "__main__":
    main()
