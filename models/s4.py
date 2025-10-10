"""
Modified from https://github.com/srush/annotated-s4
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .common import init_CV, init_VinvB, log_step_initializer, trunc_standard_normal, make_DPLR_HiPPO


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, jnp.newaxis]
    Ct = C[jnp.newaxis, :]

    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real


def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
        Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return jnp.fft.irfft(out)[: u.shape[0]]


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)

class S4Layer(nn.Module):
    Lambda_re_init: jax.Array
    Lambda_im_init: jax.Array
    P_init: jax.Array
    B_init: jax.Array

    d_state: int
    dt_min: float
    dt_max: float
    C_init: str
    l_max: int
    decode: bool = False

    def setup(self):
        # Learned Parameters (C is complex!)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (self.d_state,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (self.d_state,))
        
        # Ensure the real part of Lambda is negative (described in the SaShiMi follow-up to S4)
        self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", lambda rng, shape: self.P_init, (self.d_state,))
        self.B = self.param("B", lambda rng, shape: self.B_init, (self.d_state,))

        # Initialize state to output (C) matrix
        if self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        C = self.param("C", C_init, (self.d_state, 2))
        self.C_tilde = C[..., 0] + 1j * C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step", log_step_initializer(self.dt_min, self.dt_max), (1,))
        self.step = jnp.exp(self.log_step)

        if not self.decode:
            # CNN mode, compute kernel.
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C_tilde,
                self.step,
                self.l_max,
            )

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C_tilde,
                    self.step,
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


# Here we call vmap to parallelize across d_model
S4 = nn.vmap(
        S4Layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


def init_S4(d_state, d_model, **cfg):
    C_init = cfg.get("C_init", "complex_normal")
    dt_min = cfg.get("dt_min", 0.001)
    dt_max = cfg.get("dt_max", 0.1)
    l_max = cfg.get("seq_len", 100)
    decode = cfg.get("decode", False)


    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, P, B, _, _ = make_DPLR_HiPPO(d_state)

    # don't need to pass d_model, this is automatically handled by nn.vmap
    return partial(S4,
                   d_state=d_state,
                   Lambda_re_init=Lambda.real,
                   Lambda_im_init=Lambda.imag,
                   P_init=P,
                   B_init=B,
                   C_init=C_init,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   l_max=l_max,
                   decode=decode,
                )
