from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class Config:
    num: int = 512
    a: float = 0.4
    g: float = 5.0
    tau: float = 3.0
    tau_v: float = 144.0
    dt: float = 3.0 / 10.0
    k: float = 5.0

    @property
    def j0(self) -> float:
        return 1.0 / self.g


def wrap(x, z_range=2.0 * np.pi):
    return ((x + 0.5 * z_range) % z_range) - 0.5 * z_range


def circ_diff(x, z_range=2.0 * np.pi):
    return ((x + 0.5 * z_range) % z_range) - 0.5 * z_range


def grid(config: Config):
    return np.linspace(-np.pi, np.pi, config.num, endpoint=False)


def gauss(x, pos, amplitude, a, z_range=2.0 * np.pi):
    delta = circ_diff(x - pos, z_range=z_range)
    return amplitude * np.exp(-(delta**2) / (4.0 * a**2))


def conn(config: Config, x=None):
    x = grid(config) if x is None else np.asarray(x)
    d = circ_diff(x - x[:, None], 2.0 * np.pi)
    kernel = (
        config.j0
        / (np.sqrt(2.0 * np.pi) * config.a)
        * np.exp(-(d**2) / (2.0 * config.a**2))
    )
    return kernel


def rate(u, g, k):
    u2 = np.square(u)
    return g * u2 / (1.0 + k * np.sum(u2))


def fix_center(center, loc, a):
    center = np.where(
        (loc > np.pi - 2.0 * a) & (center < -np.pi + 2.0 * a),
        center + 2.0 * np.pi,
        center,
    )
    center = np.where(
        (center > np.pi - 2.0 * a) & (loc < -np.pi + 2.0 * a),
        center - 2.0 * np.pi,
        center,
    )
    return center


class CANN1DSFA:
    def __init__(self, config: Config, m: float, x=None, conn_mat=None):
        self.num = config.num
        self.tau = config.tau
        self.tau_v = config.tau_v
        self.g = config.g
        self.k = config.k
        self.a = config.a
        self.dt = config.dt
        self.m = m
        self.z_range = 2.0 * np.pi
        self.x = np.asarray(grid(config) if x is None else x)
        self.phase_kernel = np.exp(1j * self.x)
        self.conn_mat = conn_mat if conn_mat is not None else conn(config, self.x)

        self.u = np.zeros(config.num)
        self.v = np.zeros(config.num)
        self.input = np.zeros(config.num)
        self.center = 0.0

    def _exprel(self, x):
        if abs(x) < 1e-8:
            return 1.0 + x / 2.0
        return np.expm1(x) / x

    def step(self):
        r = rate(self.u, self.g, self.k)
        irec = np.dot(self.conn_mat, r)

        du = (-self.u + irec - self.v + self.input) / self.tau
        dv = (-self.v + self.m * self.u) / self.tau_v

        self.u[:] = np.maximum(
            self.u + self.dt * self._exprel(-self.dt / self.tau) * du, 0.0
        )
        self.v[:] = self.v + self.dt * self._exprel(-self.dt / self.tau_v) * dv
        self.input[:] = 0.0

        r = rate(self.u, self.g, self.k)
        self.center = np.angle(np.sum(r * self.phase_kernel))

    def update(self):
        self.step()
