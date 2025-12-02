import numpy as np
from abc import abstractmethod, ABC

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# ============================================================
# NUMBA IMPLEMENTATIONS
# ============================================================
if NUMBA_AVAILABLE:
    _NJIT_FAST_MATH = True


    @njit(cache=True, fastmath=_NJIT_FAST_MATH)
    def _nts_no_deadzone_numba_flat(x, k):
        eps = 1e-12
        one_minus_k = 1.0 - k
        n = x.size

        for i in range(n):
            xi = x[i]

            if xi < -1.0:
                xi = -1.0
            elif xi > 1.0:
                xi = 1.0

            ax = xi if xi >= 0 else -xi

            denom = 1.0 + k * (1.0 - 2.0 * ax)
            if denom >= 0:
                if denom < eps:
                    denom = eps
            else:
                if denom > -eps:
                    denom = -eps

            y_nts = xi * one_minus_k / denom
            y = 0.5 * (1.0 + y_nts)

            if y < 0.0:
                y = 0.0
            elif y > 1.0:
                y = 1.0

            x[i] = y


    @njit(cache=True, fastmath=_NJIT_FAST_MATH)
    def _nts_with_deadzone_numba_flat(x, k, deadzone, beta):
        eps = 1e-12
        one_minus_k = 1.0 - k
        one_minus_d = 1.0 - deadzone
        n = x.size

        for i in range(n):
            xi = x[i]

            if xi < -1.0:
                xi = -1.0
            elif xi > 1.0:
                xi = 1.0

            ax = xi if xi >= 0 else -xi
            t = ax - deadzone
            z = beta * t

            # softplus
            if z > 40.0:
                soft_dist = z / beta
            elif z < -40.0:
                soft_dist = np.exp(z) / beta
            else:
                soft_dist = np.log1p(np.exp(z)) / beta

            # normalize
            u = soft_dist / one_minus_d
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0

            sign = 1.0 if xi >= 0 else -1.0
            x_eff = sign * u

            ae = x_eff if x_eff >= 0 else -x_eff
            denom = 1.0 + k * (1.0 - 2.0 * ae)

            if denom >= 0:
                if denom < eps:
                    denom = eps
            else:
                if denom > -eps:
                    denom = -eps

            y_nts = x_eff * one_minus_k / denom
            y = 0.5 * (1.0 + y_nts)

            if y < 0.0:
                y = 0.0
            elif y > 1.0:
                y = 1.0

            x[i] = y


    @njit(cache=True, fastmath=_NJIT_FAST_MATH)
    def _power_curve_numba_flat(x, p, top):
        scale = top ** p

        for i in range(x.size):
            v = x[i]

            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0

            v = v ** p
            v /= scale

            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0

            x[i] = v


    @njit(parallel=True, fastmath=_NJIT_FAST_MATH)
    def _log1p_transform_numba_flat(x, gain, top):
        scale = np.log1p(gain * top)

        for i in range(x.size):
            v = x[i]

            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0

            v *= gain
            v = np.log1p(v)
            v /= scale

            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0

            x[i] = v


# ============================================================
# region NUMPY IMPLEMENTATIONS
# ============================================================


def _softplus_numpy(z: np.ndarray, beta: float, out: np.ndarray):
    """
    Compute out = softplus(z)/beta in-place.
    """
    neg = z < -40
    mid = z > 40  # pos
    mid |= neg    # (pos | neg)
    mid ^= True   # ~(pos | neg)

    out[:] = z                               # pos = mid = neg = z
    np.exp(out, out=out, where=(neg | mid))  # mid = neg = exp(z)
    np.log1p(out, out=out, where=mid)        # mid = log1p(exp(z))
    out /= beta                              # all divided by beta

def _nts_no_deadzone_numpy(x: np.ndarray, k: float):
    """
    In-place NTS without deadzone.
    Final values are clipped to [0,1].

    Parameters
    ----------
    x : array
        Input array (modified in-place). Expected domain ~[1,1].
        The function will clip the input at the start to enforce domain bounds.
    """
    eps = 1e-12

    np.clip(x, -1.0, 1.0, out=x)  # clamp x to [-1,1]

    # denom = 1 + k*(1 - 2|x|)   (in-place)
    denom = np.abs(x)  # |x|
    denom *= -2.0
    denom += 1.0
    denom *= k
    denom += 1.0

    # fix tiny denom in-place
    mask = np.abs(denom) < eps
    if mask.any():
        denom[mask] = np.copysign(eps, denom[mask])

    # y_nts = x*(1-k)/denom    (use x as output buffer)
    x *= (1.0 - k)
    x /= denom

    # final mapping
    x += 1.0
    x *= 0.5
    np.clip(x, 0.0, 1.0, out=x)


def _nts_with_deadzone_numpy(x: np.ndarray, k: float, deadzone: float, beta: float):
    """
    Soft deadzone version, in-place, ND.
    Final values are clipped to [0,1].

    Parameters
    ----------
    x : array
        Input array (modified in-place). Expected domain ~[1,1].
        The function will clip the input at the start to enforce domain bounds.
    """
    eps = 1e-12

    np.clip(x, -1.0, 1.0, out=x)  # clamp input domain

    ax = np.abs(x)
    t = ax - deadzone  # distance from boundary
    z = t * beta
    soft_dist = np.empty_like(z)

    _softplus_numpy(z, beta, out=soft_dist)

    # normalize into [0,1]
    u = soft_dist / (1.0 - deadzone)
    np.clip(u, 0.0, 1.0, out=u)

    x_eff = np.sign(x) * u

    # compute denom = 1 + k*(1 - 2|x_eff|)
    denom = np.abs(x_eff)
    denom *= -2.0
    denom += 1.0
    denom *= k
    denom += 1.0

    # fix tiny denom
    mask = np.abs(denom) < eps
    if mask.any():
        denom[mask] = np.copysign(eps, denom[mask])

    # compute  x_nts = x_eff * (1 - k) / denom
    x_eff *= (1.0 - k)
    x_eff /= denom
    # x_nts = x_eff

    # y = .5 * (1 + x_nts)
    x[:] = x_eff  # x_nts
    x += 1.0
    x *= 0.5
    np.clip(x, 0.0, 1.0, out=x)


def _power_curve_numpy(x: np.ndarray, p: float, top: float):
    """
    In-place clipped power curve:

        y = (x^p) / (top^p)
        final y is clipped to [0,1]

    Parameters
    ----------
    x : array
        Input array (modified in-place). Expected domain ~[0,1].
        The function will clip the input at the start to enforce domain bounds.
    """
    np.clip(x, 0.0, 1.0, out=x)  # clip input first (in case of numerical drift)
    scale = top ** p
    np.power(x, p, out=x)
    x /= scale  # final y
    np.clip(x, 0.0, 1.0, out=x)  # clip output


def _log1p_transform_numpy(x: np.ndarray, gain: float, top: float):
    """
    In-place clipped power curve:

        y = log1p(gain * x) / log1p(gain * top)
        final y is clipped to [0,1]

    Parameters
    ----------
    x : array
        Input array (modified in-place). Expected domain ~[0,1].
        The function will clip the input at the start to enforce domain bounds.
    """
    np.clip(x, 0.0, 1.0, out=x)  # clip input first (in case of numerical drift)
    x *= gain
    np.log1p(x, out=x)
    x /= np.log1p(gain * top)  # final y
    np.clip(x, 0.0, 1.0, out=x)  # clip output


# endregion

# ============================================================
# FUNC CLASSES
# ============================================================


class FuncWrapper(ABC):
    """
    Single variable function.

    inplace_func MUST be implemented, and SHOULD do computations inplace.

    Child classes should also implement get_label for plotting purposes but not required to use in texture generation.
    """

    def func(self, x: np.ndarray) -> np.ndarray:
        x_copy = x.copy()
        self.inplace_func(x_copy)
        return x_copy

    @abstractmethod
    def inplace_func(self, x: np.ndarray):
        pass


class NTSWithSoftDeadzone(FuncWrapper):
    """
    Normalized Tunable Sigmoid (NTS) with soft deadzone, in-place on ND arrays.
    credits: https://dhemery.github.io/DHE-Modules/technical/sigmoid/

    - Supports Numba (flat memory) or NumPy fallback
    - k ∈ [-0.99, 0.99]
    - Input clamped to [-1,1]
    - Output always in [0,1]
    - deadzone=0 auto-selects fast path
    """

    def __init__(self, k=0.5, deadzone=0.2, beta=0.5):
        self.k = float(k)
        self.beta = float(beta)
        self.deadzone = float(deadzone)  # uses setter

    @property
    def adjusted_beta(self):
        return (self.beta + .1) * 80 / self.deadzone

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self._beta = value

    @property
    def deadzone(self):
        return self._deadzone

    @deadzone.setter
    def deadzone(self, value):
        value = float(value)
        self._deadzone = value

        if value >= 1.0:
            raise ValueError("dead-zone must be smaller than 1")

        # choose implementation
        if value <= 0.0:
            if NUMBA_AVAILABLE:
                self._impl = lambda arr: _nts_no_deadzone_numba_flat(arr.ravel(), self.k)
            else:
                self._impl = lambda arr: _nts_no_deadzone_numpy(arr, self.k)
        else:
            if NUMBA_AVAILABLE:
                self._impl = lambda arr: _nts_with_deadzone_numba_flat(
                    arr.ravel(), self.k, value, self.adjusted_beta
                )
            else:
                self._impl = lambda arr: _nts_with_deadzone_numpy(
                    arr, self.k, value, self.adjusted_beta
                )

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        value = float(value)
        if not -0.99 <= value <= 0.99:
            raise ValueError("k must be in [-0.99, +0.99]")
        self._k = value


    def inplace_func(self, x: np.ndarray) -> None:
        """
        In-place transform on any ND C-contiguous array.
        """
        if not x.flags["C_CONTIGUOUS"]:
            raise ValueError("Input array must be C-contiguous.")

        self._impl(x)


    def get_label(self) -> str:
        dz = self.deadzone
        if dz <= 0:
            return f"NTS (k={self.k})"
        return f"NTS (Deadzone={dz:.2f}, k={self.k}, beta={self.beta})"


class PowerCurve(FuncWrapper):
    """
    Power-curve mapping in [0,1]:

        y = (clip(x, 0,1)^p) / (top^p)
        y is clamped into [0,1].

    Parameters
    ----------
    p : float
        Curvature exponent.
        - p < 1  → convex soft fade-in
        - p = 1  → identity
        - p > 1  → concave, steeper near top

    top : float
        Input at which the output should reach 1 (before clipping).
        Must satisfy 0 < top ≤ 1.

    Notes
    -----
    - Computation is performed fully in-place.
    - If Numba is available, a faster JIT version is used.
    - Input x values are clamped to [-1,1] before the transformation.
    """

    def __init__(self, p: float = 2.0, top: float = 1.0):
        if top <= 0.0 or top > 1.0:
            raise ValueError("top must be in the interval (0, 1].")

        self.p = float(p)
        self.top = float(top)

        # Choose implementation
        if NUMBA_AVAILABLE:
            self._impl = lambda arr: _power_curve_numba_flat(arr.ravel(), self.p, self.top)
        else:
            self._impl = lambda arr: _power_curve_numpy(arr, self.p, self.top)

    def inplace_func(self, x: np.ndarray):
        if not x.flags["C_CONTIGUOUS"]:
            raise ValueError("Input array must be C-contiguous for in-place transform.")

        self._impl(x)

    def get_label(self) -> str:
        return f"Power Curve (p={self.p}, top={self.top})"


class LogScalingFunc(FuncWrapper):
    """Computes log1p(x * gain) / log1p(gain * top), clipped to [0, 1]."""

    gain: float
    """Controls the steepness of the curve"""

    top: float
    """Defines the maximum input for the normalizer"""

    def __init__(self, gain: float = 100.0, top: float = .5):
        if top <= 0.0 or top > 1.0:
            raise ValueError("top must be in the interval (0, 1].")

        self.gain = float(gain)
        self.top = float(top)

        # Choose implementation
        if NUMBA_AVAILABLE:
            self._impl = lambda arr: _log1p_transform_numba_flat(arr.ravel(), self.gain, self.top)
        else:
            self._impl = lambda arr: _log1p_transform_numpy(arr, self.gain, self.top)

    def inplace_func(self, x: np.ndarray):
        if not x.flags["C_CONTIGUOUS"]:
            raise ValueError("Input array must be C-contiguous for in-place transform.")

        self._impl(x)

    def get_label(self) -> str:
        return f'Log Scaling (Gain={self.gain:.2f}, Top={self.top:.2f})'
