from math import sqrt
import abc
import jax

from jax import lax
from flax import linen as nn

from jax import numpy as jnp
from jax.nn.initializers import normal, zeros
from netket.jax.utils import dtype_complex, dtype_real
from plum import dispatch

# from .mps import MPS, _update_h_p_single

from .gpu_cond import gpu_cond
from .reorder import get_reorder_idx, get_reorder_prev, inv_reorder, reorder
from .symmetry import symmetrize_model

from jax.scipy.linalg import eigh
from netket.utils.types import Array, DType, NNInitFunc, PRNGKeyT, PyTree
from netket.hilbert.homogeneous import HomogeneousHilbert
from flax.core.frozen_dict import unfreeze, freeze

class AbstractARNN(nn.Module):
    """
    Base class for autoregressive neural networks.

    Subclasses must implement the method `conditionals_log_psi`, or override the methods
    `__call__` and `conditionals` if desired.

    They can override `conditional` to implement the caching for fast autoregressive sampling.
    See :class:`netket.nn.FastARNNConv1D` for example.

    They must also implement the field `machine_pow`,
    which specifies the exponent to normalize the outputs of `__call__`.
    """

    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous unconstrained Hilbert spaces are supported."""

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.hilbert, HomogeneousHilbert):
            raise ValueError(
                f"Only homogeneous Hilbert spaces are supported by ARNN, but hilbert is a {type(self.hilbert)}."
            )

        if self.hilbert.constrained:
            raise ValueError("Only unconstrained Hilbert spaces are supported by ARNN.")

    @property
    def _use_naive_scan(self) -> bool:
        """
        Use a naive version of `jax.lax.scan` in `ARDirectSampler`.
        It should be True if `_conditional` is not compatible with the jitted `scan`.
        """
        return False

    @abc.abstractmethod
    def conditionals_log_psi(self, inputs: Array) -> Array:
        """
        Computes the log of the conditional wave-functions for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The log psi with dimensions (batch, Hilbert.size, Hilbert.local_size).
        """

    def conditionals(self, inputs: Array) -> Array:
        """
        Computes the conditional probabilities for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).

        Examples:

          >>> import pytest; pytest.skip("skip automated test of this docstring")
          >>>
          >>> p = model.apply(variables, σ, method=model.conditionals)
          >>> print(p[2, 3, :])
          [0.3 0.7]
          # For the 3rd spin of the 2nd sample in the batch,
          # it takes probability 0.3 to be spin down (local state index 0),
          # and probability 0.7 to be spin up (local state index 1).
        """
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        log_psi = self.conditionals_log_psi(inputs)

        p = jnp.exp(self.machine_pow * log_psi.real)
        return p

    def conditional(self, inputs: Array, index: int) -> Array:
        """
        Computes the conditional probabilities for one site to take each value.

        It should only be called successively with indices 0, 1, 2, ...,
        as in the autoregressive sampling procedure.

        Args:
          inputs: configurations of partially sampled sites with dimensions (batch, Hilbert.size),
            where the sites that `index` depends on must be already sampled.
          index: index of the site being queried.

        Returns:
          The probabilities with dimensions (batch, Hilbert.local_size).
        """
        # TODO: remove this in future
        if hasattr(self, "_conditional"):
            from netket.utils import warn_deprecation

            warn_deprecation(
                "AbstractARNN._conditional has been renamed to AbstractARNN.conditional "
                "as a public API. Please update your subclass to use fast AR sampling."
            )
            return self._conditional(inputs, index)

        return self.conditionals(inputs)[:, index, :]

    def __call__(self, inputs: Array) -> Array:
        """
        Computes the log wave-functions for input configurations.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The log psi with dimension (batch,).
        """

        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        idx = self.hilbert.states_to_local_indices(inputs)
        idx = jnp.expand_dims(idx, axis=-1)

        log_psi = self.conditionals_log_psi(inputs)

        log_psi = jnp.take_along_axis(log_psi, idx, axis=-1)
        log_psi = log_psi.reshape((inputs.shape[0], -1)).sum(axis=1)
        return log_psi

    def _init_independent_cache(self, inputs: Array) -> None:
        self.conditional(inputs, 0)

    def _init_dependent_cache(self, inputs: Array) -> None:
        pass

    # --------- CHANGED! ----------------------------------------------------------------
    def init_cache(self, variables: PyTree, inputs: Array, key: PRNGKeyT) -> PyTree:
        variables_tmp = self.init(key, inputs, method=self._init_independent_cache)
        cache_prev = variables.get("cache")
        cache = variables_tmp.get("cache")

        if cache:
            variables = {**variables, "cache": cache}

        _, mutables = self.apply(
            variables, inputs, method=self._init_dependent_cache, mutable=["cache"]
        )

        cache = unfreeze(mutables.get("cache"))
        cache['progress'] = cache_prev['progress']
        cache = freeze(cache)

        return cache
    # -----------------------------------------------------------------------------------

    def reorder(self, inputs: Array, axis: int = 0) -> Array:
        """
        Transforms an array from unordered to ordered.

        We call a 1D array 'unordered' if we need non-trivial indexing to access
        its elements in the autoregressive order, e.g., `a[0], a[1], a[3], a[2]`
        for the snake ordering. Otherwise, we call it 'ordered'.

        The inputs of `conditionals_log_psi`, `conditionals`, `conditional`, and
        `__call__` are assumed to have unordered layout.
        """
        return inputs

    def inverse_reorder(self, inputs: Array, axis: int = 0) -> Array:
        """
        Transforms an array from ordered to unordered. See `reorder`.
        """
        return inputs


def canonize_mps(M, *, eps=1e-15):
    def scan_func(_, m):
        mm = jnp.einsum("iab,iac->bc", m.conj(), m)
        lam, u = eigh(mm)
        u /= jnp.sqrt(jnp.abs(lam)) + eps
        m = jnp.einsum("iab,bc->iac", m, u)
        return None, m

    _, M = lax.scan(scan_func, None, M)
    return M


def norm_mps(M, left_boundary, right_boundary, reorder_idx):
    def scan_func(p, m):
        p = jnp.einsum("ab,iac,ibd->cd", p, m.conj(), m)
        return p, None

    p = jnp.einsum("a,b->ab", left_boundary.conj(), left_boundary)
    M = M[reorder_idx]
    p, _ = lax.scan(scan_func, p, M)
    p = jnp.einsum("ab,a,b->", p, right_boundary.conj(), right_boundary).real
    return p


def wrap_M_init_canonize(M_init, left_boundary, right_boundary, reorder_idx):
    def wrapped_M_init(*args):
        M = M_init(*args)
        L = M.shape[0]
        M = canonize_mps(M)
        p = norm_mps(M, left_boundary, right_boundary, reorder_idx)
        M = M * p ** (-1 / (2 * L))
        return M

    return wrapped_M_init


def get_gamma(M, right_boundary, reorder_idx=None, inv_reorder_idx=None):
    def scan_func(gamma_old, m):
        gamma = jnp.einsum("iab,icd,bd->ac", m.conj(), m, gamma_old)
        return gamma, gamma_old

    gamma_L = jnp.einsum("a,b->ab", right_boundary.conj(), right_boundary)
    if reorder_idx is not None:
        M = M[reorder_idx]
    _, gamma = lax.scan(scan_func, gamma_L, M, reverse=True)
    if inv_reorder_idx is not None:
        gamma = gamma[inv_reorder_idx]
    return gamma


class MPSRNN1D_local(AbstractARNN):
    bond_dim: int
    zero_mag: bool
    refl_sym: bool
    affine: bool
    nonlin: bool
    no_phase: bool
    no_w_phase: bool
    cond_psi: bool
    reorder_type: str
    reorder_dim: int
    dtype: DType = jnp.complex64
    machine_pow: int = 2
    eps: float = 1e-7

    def _common_setup(self):
        B = self.bond_dim

        self.left_boundary = jnp.ones((B,), dtype=self.dtype)
        self.right_boundary = jnp.ones((B,), dtype=self.dtype)

        self.reorder_idx, self.inv_reorder_idx = get_reorder_idx(
            self.reorder_type, self.reorder_dim, self.hilbert.size
        )
        self.reorder_prev = get_reorder_prev(self.reorder_idx, self.inv_reorder_idx)

        self.h = self.variable("cache", "h", lambda: None)
        self.counts = self.variable("cache", "counts", lambda: None)
        self.progress = self.variable("cache","progress", jnp.zeros, (1,), jnp.float32)

    def _get_gamma(self):
        raise NotImplementedError

    def setup(self):
        self._common_setup()

        L = self.hilbert.size
        S = self.hilbert.local_size
        B = self.bond_dim

        M_init = normal(stddev=1 / sqrt(B))
        self.M = self.param("M", M_init, (L, S, B, B), self.dtype)
        
        if self.affine:
            v_init = normal(stddev=1)
            self.v = self.param("v", v_init, (L, S, B), self.dtype)

        if not self.no_phase and not self.no_w_phase:
            if self.cond_psi:
                self.w_phase = self.param(
                    "w_phase", normal(stddev=1), (L, B), self.dtype
                )
                self.c_phase = self.param("c_phase", zeros, (L,), self.dtype)
            else:
                self.w_phase = self.param("w_phase", normal(stddev=1), (B,), self.dtype)
                self.c_phase = self.param("c_phase", zeros, (), self.dtype)

        self.log_gamma = self.param("log_gamma", zeros, (L, B), dtype_real(self.dtype))

    def _init_independent_cache(self, inputs):
        S = self.hilbert.local_size
        B = self.bond_dim

        batch_size = inputs.shape[0]
        self.h.value = jnp.full((batch_size, S, B), self.left_boundary)
        self.counts.value = jnp.zeros((batch_size, S), dtype=jnp.int32)
        # self.progress.value = jnp.zeros((1,),dtype=jnp.float32)

    def _init_dependent_cache(self, _):
        pass

    def _preprocess_dim(self, inputs):
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        return inputs

    def _conditionals(self, inputs, index):
        inputs = self._preprocess_dim(inputs)
        p, self.h.value, self.counts.value = _update_h_p(
            self, inputs, index, self.h.value, self.counts.value
        )
        return p

    def conditionals(self, inputs):
        inputs = self._preprocess_dim(inputs)
        p = _conditionals(self, inputs)
        return p

    def __call__(self, inputs): 
        inputs = self._preprocess_dim(inputs)
        if self.refl_sym:
            return symmetrize_model(lambda x: _call(self, x))(inputs)
        else:
            return _call(self, inputs)

    def reorder(self, inputs):
        return reorder(self, inputs)

    def inverse_reorder(self, inputs):
        return inv_reorder(self, inputs)  
      
    def _nonlin_fxn(self,preact):
        return 1/(1+jnp.exp(-preact))
    
    def _map_prog(self,progress):
        # progress = jnp.round(progress,decimals=1)
        return 1/(1+jnp.exp(-10*(progress-0.5)))

    def _get_new_h(self, h, i):
        h = jnp.einsum("a,iab->ib", h, self.M[i])
        if self.affine:
            h += self.v[i]
        if self.nonlin:
            # Dynamic non-linearity
            coeff = self.progress.value
            coeff = self._map_prog(coeff)
            return (1. - coeff) * h + coeff * self._nonlin_fxn(h)
        else:
            return h

@dispatch
def _get_p(model: MPSRNN1D_local, h, i):
    return jnp.einsum("ia,ia,a->i", h.conj(), h, jnp.exp(model.log_gamma[i])).real


@dispatch
def _call_single(model: MPSRNN1D_local, inputs):
    qn = model.hilbert.states_to_local_indices(inputs)

    def scan_func(carry, i):
        h, log_psi, counts = carry
        p_i, h, counts = _update_h_p_single(model, inputs, i, h, counts)
        p_i /= p_i.sum()
        p_i = p_i[qn[i]]
        log_psi += jnp.log(p_i) / 2

        if not model.no_phase and model.cond_psi:
            if model.no_w_phase:
                phi = h[qn[i]] @ model.right_boundary
            else:
                phi = h[qn[i]] @ model.w_phase[i] + model.c_phase[i]
            log_psi += jnp.angle(phi) * 1j

        return (h, log_psi, counts), None

    S = model.hilbert.local_size
    B = model.bond_dim

    h = jnp.full((S, B), model.left_boundary)
    if model.no_phase:
        log_psi = jnp.zeros((), dtype=dtype_real(model.dtype))
    else:
        log_psi = jnp.zeros((), dtype=jnp.complex128)
    counts = jnp.zeros((S,), dtype=jnp.int32)

    (h, log_psi, _), _ = lax.scan(scan_func, (h, log_psi, counts), model.reorder_idx)

    if not model.no_phase and not model.cond_psi:
        i = model.reorder_idx[-1]
        if model.no_w_phase:
            phi = h[qn[i]] @ model.right_boundary
        else:
            phi = h[qn[i]] @ model.w_phase + model.c_phase
        log_psi += jnp.angle(phi) * 1j

    return log_psi


# ----------------------------------------------------------------------------------

def _normalize_h(h):
    h /= jnp.sqrt((h.conj() * h).real.mean())
    return h

def _update_h_p_single(model, inputs, i, h, counts):
    L = model.hilbert.size
    qn = model.hilbert.states_to_local_indices(inputs)

    qn_i = qn[model.reorder_prev[i]]
    h = h[qn_i]
    h = model._get_new_h(h, i)
    h = _normalize_h(h)

    p = _get_p(model, h, i)

    counts = gpu_cond(
        i != model.reorder_idx[0],
        lambda _: counts.at[qn_i].add(1),
        lambda _: counts,
        None,
    )
    if model.zero_mag:
        p = jnp.where(counts < L // 2, p, model.eps)

    return p, h, counts


# inputs: (batch_size, L)
# h: (batch_size, S, B)
_update_h_p = jax.vmap(_update_h_p_single, in_axes=(None, 0, None, 0, 0))


def _conditionals_single(model, inputs):
    def scan_func(carry, i):
        h, p, counts = carry
        p_i, h, counts = _update_h_p_single(model, inputs, i, h, counts)
        p = p.at[i].set(p_i)
        return (h, p, counts), None

    L = model.hilbert.size
    S = model.hilbert.local_size
    B = model.bond_dim

    h = jnp.full((S, B), model.left_boundary)
    p = jnp.empty((L, S), dtype=dtype_real(model.dtype))
    counts = jnp.zeros((S,), dtype=jnp.int32)
    (_, p, _), _ = lax.scan(scan_func, (h, p, counts), model.reorder_idx)
    return p


# inputs: (batch_size, L)
_conditionals = jax.vmap(_conditionals_single, in_axes=(None, 0))

# inputs: (batch_size, L)
_call = jax.vmap(_call_single, in_axes=(None, 0))
