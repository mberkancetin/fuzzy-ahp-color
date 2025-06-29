from __future__ import annotations
import numpy as np
from typing import Protocol, TypeVar, Union, Any, Dict, List, Callable, runtime_checkable
from functools import total_ordering

# ==============================================================================
# 1. THE PROTOCOL BLUEPRINT
# ==============================================================================

@runtime_checkable
class NumericType(Protocol):
    """
    A protocol defining the interface for any numeric type used in the AHP library.
    This is calibrated to match the structure of the provided TFN class.
    """

    # Standard Operators
    def __add__(self, other: Union['NumericType', float]) -> 'NumericType': ...
    def __sub__(self, other: Union['NumericType', float]) -> 'NumericType': ...
    def __mul__(self, other: Union['NumericType', float]) -> 'NumericType': ...
    def __truediv__(self, other: Union['NumericType', float]) -> 'NumericType': ...
    def __pow__(self, exponent: float) -> 'NumericType': ...

    # Reflected Operators
    def __radd__(self, other: float) -> 'NumericType': ...
    def __rsub__(self, other: float) -> 'NumericType': ...
    def __rmul__(self, other: float) -> 'NumericType': ...
    def __rtruediv__(self, other: float) -> 'NumericType': ...

    # Comparison Operators (for @total_ordering)
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: Union['NumericType', float]) -> bool: ...

    # Utility Methods
    def inverse(self) -> 'NumericType': ...
    @staticmethod
    def neutral_element() -> 'NumericType': ...
    @staticmethod
    def multiplicative_identity() -> 'NumericType': ...
    @staticmethod
    def from_crisp(value: float) -> 'NumericType': ...
    def power(self, exponent: float) -> 'NumericType': ...
    def alpha_cut(self, alpha: float) -> tuple[float, float]: ...
    def defuzzify(self, method: str = 'centroid', **kwargs) -> float: ...
    @classmethod
    def get_available_defuzzify_methods(cls) -> List[str]: ...
    @classmethod
    def register_defuzzify_method(cls, name: str, func: Callable): ...

# ==============================================================================
# 2. IMPLEMENTATIONS OF THE PROTOCOL
# ==============================================================================

@total_ordering
class Crisp:
    """
    A wrapper for float to make it conform to the NumericType protocol.
    """
    _defuzzify_methods: Dict[str, Callable] = {}

    def __init__(self, value: float):
        """
        Initializes a Crisp number.
        This constructor is idempotent: Crisp(Crisp(5.0)) is valid.
        """
        if isinstance(value, Crisp):
            self.value = value.value
        else:
            self.value = float(value)

    def __repr__(self) -> str:
        return f"Crisp({self.value:.4f})"

    def _get_value(self, other: Union[Crisp, float]) -> float:
        """Helper to extract the float value from another object."""
        return other.value if isinstance(other, Crisp) else float(other)

    def __add__(self, other: Union[Crisp, float]) -> Crisp:
        return Crisp(self.value + self._get_value(other))

    def __radd__(self, other: float) -> Crisp:
        return self.__add__(other)

    def __sub__(self, other: Union[Crisp, float]) -> Crisp:
        return Crisp(self.value - self._get_value(other))

    def __rsub__(self, other: float) -> Crisp:
        return Crisp(self._get_value(other) - self.value)

    def __mul__(self, other: Union[Crisp, float]) -> Crisp:
        return Crisp(self.value * self._get_value(other))

    def __rmul__(self, other: float) -> Crisp:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Crisp, float]) -> Crisp:
        val = self._get_value(other)
        if val == 0: raise ZeroDivisionError("Division by zero.")
        return Crisp(self.value / val)

    def __rtruediv__(self, other: float) -> Crisp:
        if self.value == 0: raise ZeroDivisionError("Division by zero.")
        return Crisp(self._get_value(other) / self.value)

    def __pow__(self, exponent: float) -> Crisp:
        return Crisp(self.value ** exponent)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Crisp): return self.value == other.value
        if isinstance(other, (int, float)): return self.value == other
        return False

    def __lt__(self, other: Union[Crisp, float]) -> bool:
        return self.value < self._get_value(other)

    def inverse(self) -> Crisp:
        if self.value == 0: raise ValueError("Cannot invert zero.")
        return Crisp(1.0 / self.value)

    @staticmethod
    def neutral_element() -> Crisp:
        return Crisp(0.0)

    @staticmethod
    def multiplicative_identity() -> Crisp:
        return Crisp(1.0)

    @staticmethod
    def from_crisp(value: float) -> Crisp:
        return Crisp(value)

    def power(self, exponent: float) -> Crisp:
        return self.__pow__(exponent)

    def get_crisp_value(item: Any, method: str = 'centroid') -> float:
        """Safely gets the float value from a NumericType or a primitive number."""
        if hasattr(item, 'value'): # It's our Crisp class
            return item.value
        elif hasattr(item, 'defuzzify'): # It's a TFN, TrFN, etc.
            return item.defuzzify(method=method)
        elif isinstance(item, (int, float)): # It's already a primitive number
            return float(item)
        else:
            raise TypeError(f"Cannot extract a crisp value from type {type(item)}")

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        # For a crisp number, the interval is just the number itself for any alpha > 0
        return (self.value, self.value)

    def defuzzify(self, method: str = 'centroid', **kwargs) -> float:
        func = Crisp._defuzzify_methods.get(method)
        if func is None:
            fixed_func = lambda _: self.value
            self.register_defuzzify_method(name=method, func=fixed_func)
            func = Crisp._defuzzify_methods.get(method)
        return func(self, **kwargs)

    @classmethod
    def get_available_defuzzify_methods(cls) -> List[str]:
        return list(cls._defuzzify_methods.keys())

    @classmethod
    def register_defuzzify_method(cls, name: str, func: Callable):
        """Registers a new defuzzification function for this number type."""
        if name in cls._defuzzify_methods:
            print(f"Warning: Overwriting defuzzify method '{name}' for {cls.__name__}")
        cls._defuzzify_methods[name] = func


@total_ordering
class TFN:
    """
    Triangular Fuzzy Number (TFN) class.
    A TFN is represented as (l, m, u) where l ≤ m ≤ u.
    l: lower bound, m: middle value, u: upper bound
    """
    _defuzzify_methods: Dict[str, Callable] = {}

    def __init__(self, l, m, u):
        """Initialize a triangular fuzzy number."""
        if not (l <= m <= u):
            raise ValueError("TFN values must satisfy l <= m <= u")
        self.l = float(l)
        self.m = float(m)
        self.u = float(u)

    def __repr__(self):
        """String representation of the TFN."""
        return f"TFN({self.l:.4f}, {self.m:.4f}, {self.u:.4f})"

    def _get_other_as_tfn(self, other: Union[TFN, Crisp, float]) -> TFN:
        if isinstance(other, TFN): return other
        val = other.value if hasattr(other, 'value') else float(other)
        return TFN(val, val, val)

    def __float__(self):
        """Convert the custom class instance to a middle float value"""
        return float(self.m)

    def __add__(self, other: Union[TFN, Crisp, float]) -> TFN:
        o = self._get_other_as_tfn(other)
        return TFN(self.l + o.l, self.m + o.m, self.u + o.u)

    def __radd__(self, other: float) -> TFN:
        return self.__add__(other)

    def __sub__(self, other: Union[TFN, Crisp, float]) -> TFN:
        o = self._get_other_as_tfn(other)
        return TFN(self.l - o.u, self.m - o.m, self.u - o.l)

    def __rsub__(self, other: float) -> TFN:
        o = self._get_other_as_tfn(other)
        return o - self

    def __mul__(self, other: Union[TFN, Crisp, float]) -> TFN:
        o = self._get_other_as_tfn(other)
        return TFN(self.l * o.l, self.m * o.m, self.u * o.u)

    def __rmul__(self, other: float) -> TFN:
        return self.__mul__(other)

    def __truediv__(self, other: Union[TFN, Crisp, float]) -> TFN:
        """Calculates self / other using the standard fuzzy arithmetic rule."""
        if isinstance(other, TFN):
            if other.l <= 0:
                raise ZeroDivisionError("Cannot divide by a TFN whose range includes zero or negative numbers.")
            return TFN(self.l / other.u, self.m / other.m, self.u / other.l)
        elif isinstance(other, (int, float)):
            if other == 0: raise ZeroDivisionError
            # For division by a scalar, the order is preserved if positive
            if other > 0:
                return TFN(self.l / other, self.m / other, self.u / other)
            else: # If dividing by a negative number, the bounds flip
                return TFN(self.u / other, self.m / other, self.l / other)
        elif hasattr(other, 'value'):
            return self.__truediv__(other.value)
        return NotImplemented

    def __rtruediv__(self, other: float) -> TFN:
        o = self._get_other_as_tfn(other)
        return o / self

    def __pow__(self, exponent: float) -> TFN:
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        if exponent >= 0:
            return TFN(self.l**exponent, self.m**exponent, self.u**exponent)
        return self.inverse().__pow__(-exponent)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TFN):
            return self.l == other.l and self.m == other.m and self.u == other.u
        return False

    def __lt__(self, other: Union[TFN, Crisp, float]) -> bool:
        other_centroid = other.defuzzify() if hasattr(other, 'centroid') else float(other)
        return self.defuzzify() < other_centroid

    def inverse(self) -> TFN:
        """Return the inverse of the TFN."""
        # Ensure none of the values are zero to avoid division by zero
        if self.l <= 0:
            raise ValueError("Cannot invert a TFN with non-positive values")
        return TFN(1.0/self.u, 1.0/self.m, 1.0/self.l)

    @staticmethod
    def neutral_element() -> TFN:
        return TFN(0.0, 0.0, 0.0)

    @staticmethod
    def multiplicative_identity() -> TFN:
        return TFN(1.0, 1.0, 1.0)

    @staticmethod
    def from_crisp(value: float) -> TFN:
        return TFN(value, value, value)

    def power(self, exponent: float) -> TFN:
        return self.__pow__(exponent)

    # Utility methods
    def to_array(self):
        """Convert to NumPy array."""
        return np.array([self.l, self.m, self.u])

    def to_float(self):
        return np.sum([self.l, self.m, self.u])/3

    def distance(self, other):
        """
        Calculate the distance between two TFNs.
        Using the vertex method.
        """
        if not isinstance(other, TFN):
            raise TypeError("Can only calculate distance between two TFNs")

        d_l = (self.l - other.l)**2
        d_m = (self.m - other.m)**2
        d_u = (self.u - other.u)**2

        return np.sqrt((d_l + d_m + d_u) / 3.0)

    def possibility_degree(self, other: TFN) -> float:
        """
        Calculate the possibility degree that self >= other.
        V(self >= other), used in Chang's extent analysis method.
        """
        if self.m >= other.m:
            return 1.0
        if other.l >= self.u:
            return 0.0
        denominator = (self.m - self.u) - (other.m - other.l)
        if abs(denominator) < 1e-9:
            return 1.0
        return (other.l - self.u) / denominator

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        if not (0 <= alpha <= 1): raise ValueError("Alpha must be between 0 and 1.")
        lower = self.l + alpha * (self.m - self.l)
        upper = self.u - alpha * (self.u - self.m)
        return lower, upper

    def defuzzify(self, method: str = 'graded_mean', **kwargs) -> float:
        """
        Defuzzifies the TFN by dispatching to a registered method.

        .. note::
            For TFNs, 'graded_mean' (l+4m+u)/6 is often preferred as it is a
            well-regarded method (e.g., Yager's approach) that considers all
            points of the fuzzy number. 'centroid' (l+m+u)/3 is simpler.
        """
        func = self.__class__._defuzzify_methods.get(method)
        if func is None:
            available = list(self.__class__._defuzzify_methods.keys())
            raise ValueError(f"Method '{method}' not implemented for TFN. Available: {available}")
        return func(self, **kwargs)

    @classmethod
    def get_available_defuzzify_methods(cls) -> List[str]:
        return list(cls._defuzzify_methods.keys())

    @classmethod
    def register_defuzzify_method(cls, name: str, func: Callable):
        """Registers a new defuzzification function for this number type."""
        if name in cls._defuzzify_methods:
            print(f"Warning: Overwriting defuzzify method '{name}' for {cls.__name__}")
        cls._defuzzify_methods[name] = func


@total_ordering
class TrFN:
    """
    Implementation of a Trapezoidal Fuzzy Number (a, b, c, d).
    """
    _defuzzify_methods: Dict[str, Callable] = {}

    def __init__(self, a: float, b: float, c: float, d: float):
        if not a <= b <= c <= d:
            raise ValueError(f"TrFN values must be in order a<=b<=c<=d, but got a={a}, b={b}, c={c}, d={d}")
        self.a, self.b, self.c, self.d = float(a), float(b), float(c), float(d)

    def __repr__(self) -> str:
        return f"TrFN({self.a:.4f}, {self.b:.4f}, {self.c:.4f}, {self.d:.4f})"

    def _get_other_as_trfn(self, other: Union[TrFN, Crisp, float]) -> TrFN:
        if isinstance(other, TrFN):
            return other
        val = other.value if hasattr(other, 'value') else float(other)
        return TrFN(val, val, val, val)

    def __add__(self, other: Union[TrFN, Crisp, float]) -> TrFN:
        o = self._get_other_as_trfn(other)
        return TrFN(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    def __radd__(self, other: float) -> TrFN:
        return self.__add__(other)

    def __sub__(self, other: Union[TrFN, Crisp, float]) -> TrFN:
        o = self._get_other_as_trfn(other)
        return TrFN(self.a - o.d, self.b - o.c, self.c - o.b, self.d - o.a)

    def __rsub__(self, other: float) -> TrFN:
        o = self._get_other_as_trfn(other)
        return o - self

    def __mul__(self, other: Union[TrFN, float]) -> TrFN:
        if isinstance(other, TrFN):
            return TrFN(self.a * other.a, self.b * other.b, self.c * other.c, self.d * other.d)
        else: # Scalar multiplication
            val = float(other)
            if val >= 0:
                return TrFN(self.a * val, self.b * val, self.c * val, self.d * val)
            else:
                return TrFN(self.d * val, self.c * val, self.b * val, self.a * val)

    def __mul__(self, other: Union[TrFN, Crisp, float]) -> TrFN:
        o = self._get_other_as_trfn(other)
        return TrFN(self.a * o.a, self.b * o.b, self.c * o.c, self.d * o.d)

    def __rmul__(self, other: float) -> TrFN:
        return self.__mul__(other)

    def __truediv__(self, other: Union[TrFN, Crisp, float]) -> TrFN:
        if isinstance(other, TrFN):
            if other.a <= 0: raise ZeroDivisionError(...)
            return TrFN(self.a / other.d, self.b / other.c, self.c / other.b, self.d / other.a)

    def __rtruediv__(self, other: float) -> TrFN:
        o = self._get_other_as_trfn(other)
        return o / self

    def __pow__(self, exponent: float) -> TrFN:
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        if exponent >= 0:
            return TrFN(self.a**exponent, self.b**exponent, self.c**exponent, self.d**exponent)
        return self.inverse().__pow__(-exponent)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrFN):
            return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d
        return False

    def __lt__(self, other: Union[TrFN, Crisp, float]) -> bool:
        other_centroid = other.defuzzify() if hasattr(other, 'centroid') else float(other)
        return self.defuzzify() < other_centroid

    def inverse(self) -> TrFN:
        if self.a <= 0:
            raise ValueError("Cannot invert a TrFN with non-positive values.")
        return TrFN(1.0/self.d, 1.0/self.c, 1.0/self.b, 1.0/self.a)

    @staticmethod
    def multiplicative_identity() -> TrFN:
        return TrFN(1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def neutral_element() -> TrFN:
        return TrFN(0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_crisp(value: float) -> TrFN:
        return TrFN(value, value, value, value)

    @staticmethod
    def from_tfn(tfn: TFN) -> TrFN:
        """Converts a TFN to a degenerate TrFN."""
        return TrFN(tfn.l, tfn.m, tfn.m, tfn.u)

    def power(self, exponent: float) -> TrFN:
        return self.__pow__(exponent)

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        if not (0 <= alpha <= 1): raise ValueError("Alpha must be between 0 and 1.")
        # Similar logic for a trapezoid
        lower = self.a + alpha * (self.b - self.a)
        upper = self.d - alpha * (self.d - self.c)
        return lower, upper

    def defuzzify(self, method: str = 'centroid', **kwargs) -> float:
        """
        Defuzzifies the TrFN by dispatching to a registered method.
        """
        func = self.__class__._defuzzify_methods.get(method)
        if func is None:
            available = list(self.__class__._defuzzify_methods.keys())
            raise ValueError(f"Method '{method}' not implemented for TrFN. Available: {available}")
        return func(self, **kwargs)

    @classmethod
    def get_available_defuzzify_methods(cls) -> List[str]:
        return list(cls._defuzzify_methods.keys())

    @classmethod
    def register_defuzzify_method(cls, name: str, func: Callable):
        """Registers a new defuzzification function for this number type."""
        if name in cls._defuzzify_methods:
            print(f"Warning: Overwriting defuzzify method '{name}' for {cls.__name__}")
        cls._defuzzify_methods[name] = func


@total_ordering
class IFN:
    """
    Implementation of an Intuitionistic Fuzzy Number (μ, ν), representing
    degrees of membership, non-membership, and hesitation (π = 1 - μ - ν).

    .. note::
        **Academic Note:** Intuitionistic Fuzzy Sets (IFS) by Atanassov (1986)
        extend fuzzy sets by allowing for explicit modeling of hesitation. The
        arithmetic operations implemented here are based on the standard IFS
        algebra as defined by Atanassov and others.
    """
    _defuzzify_methods: Dict[str, Callable] = {}

    def __init__(self, mu: float, nu: float):
        mu, nu = float(mu), float(nu)
        if not (0 <= mu <= 1 and 0 <= nu <= 1):
            raise ValueError("Membership (mu) and non-membership (nu) must be between 0 and 1.")
        if round(mu + nu, 9) > 1.0: # Use round to handle float precision issues
            raise ValueError(f"Sum of membership and non-membership must not exceed 1, but mu+nu={mu+nu}.")
        self.mu = mu
        self.nu = nu
        self.pi = 1.0 - self.mu - self.nu

    def __repr__(self) -> str:
        return f"IFN(μ={self.mu:.4f}, ν={self.nu:.4f})"

    def _get_other_as_ifn(self, other: Union[IFN, Crisp, float]) -> IFN:
        """Helper to convert other types to IFN for operations."""
        if isinstance(other, IFN): return other
        # A crisp number has no hesitation, so ν = 1 - μ
        val = other.value if hasattr(other, 'value') else float(other)
        if not (0 <= val <= 1):
            raise ValueError("Cannot perform arithmetic with crisp value outside [0,1] against an IFN.")
        return IFN(mu=val, nu=1-val)

    # --- Arithmetic Operations ---
    def __add__(self, other: IFN) -> IFN:
        if not isinstance(other, IFN): return NotImplemented
        return IFN(
            mu = self.mu + other.mu - self.mu * other.mu,
            nu = self.nu * other.nu
        )

    def __mul__(self, other: IFN) -> IFN:
        if not isinstance(other, IFN): return NotImplemented
        return IFN(
            mu = self.mu * other.mu,
            nu = self.nu + other.nu - self.nu * other.nu
        )

    # Subtraction and Division are not standardly defined for IFS in a way that
    # is useful for AHP. They are often context-specific or not used at all.
    # Returning NotImplemented is the correct, safe approach.
    def __sub__(self, other): return NotImplemented
    def __truediv__(self, other): return NotImplemented

    # --- Reflected and Scalar Operations ---
    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __rsub__(self, other): return NotImplemented
    def __rtruediv__(self, other): return NotImplemented

    def scale(self, scalar: float) -> IFN:
        """Scalar multiplication (λ * A), a key IFS operation."""
        if not (0 <= scalar):
            raise ValueError("Scalar for IFN multiplication must be non-negative.")
        return IFN(
            mu = 1 - (1 - self.mu) ** scalar,
            nu = self.nu ** scalar
        )

    def power(self, exponent: float) -> IFN:
        """Raises the IFN to a scalar power (A ^ λ)."""
        if not (0 <= exponent):
            raise ValueError("Exponent for IFN must be non-negative.")
        return IFN(
            mu = self.mu ** exponent,
            nu = 1 - (1 - self.nu) ** exponent
        )

    def __pow__(self, exponent: float) -> IFN:
        return self.power(exponent)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IFN):
            return self.mu == other.mu and self.nu == other.nu
        return False

    def __lt__(self, other: IFN) -> bool:
        if not isinstance(other, IFN): return NotImplemented
        # Standard comparison rule for IFNs
        if self.defuzzify(method="score") < other.defuzzify(method="score"):
            return True
        elif self.defuzzify(method="score") == other.defuzzify(method="score"):
            return self.defuzzify(method="accuracy")  < other.defuzzify(method="accuracy")
        return False

    def inverse(self) -> IFN:
        """The inverse (or complement) of an IFN is (ν, μ)."""
        return IFN(mu=self.nu, nu=self.mu)

    def hesitancy(self) -> float:
        """
        Returns the hesitancy degree (π = 1 - μ - ν), also known as the
        intuitionistic fuzzy index. A higher value indicates more uncertainty.
        """
        return self.pi

    @staticmethod
    def neutral_element() -> IFN:
        """The additive identity for IFN: (0, 1), representing 'false' or full non-membership."""
        return IFN(0.0, 1.0)

    @staticmethod
    def multiplicative_identity() -> IFN:
        """The multiplicative identity for IFN: (1, 0), representing 'true' or full membership."""
        return IFN(1.0, 0.0)

    @staticmethod
    def from_crisp(value: float) -> IFN:
        """Creates an IFN from a crisp value [0,1], assuming no hesitation."""
        if not (0 <= value <= 1):
            raise ValueError("Crisp value for IFN conversion must be between 0 and 1.")
        return IFN(mu=value, nu=1.0 - value)

    def alpha_cut(self, alpha: float): return NotImplemented

    def defuzzify(self, method: str = 'score', **kwargs) -> float:
        """
        Defuzzifies the IFN into a crisp value using various registered methods.

        Args:
            method: 'centroid', 'score', 'entropy', 'accuracy', 'value'.
        """
        func = self.__class__._defuzzify_methods.get(method)
        if func is None:
            available = list(self.__class__._defuzzify_methods.keys())
            raise ValueError(f"Method '{method}' not implemented for IFN. Available: {available}")
        return func(self, **kwargs)

    @classmethod
    def get_available_defuzzify_methods(cls) -> List[str]:
        return list(cls._defuzzify_methods.keys())

    @classmethod
    def register_defuzzify_method(cls, name: str, func: Callable):
        """Registers a new defuzzification function for this number type."""
        if name in cls._defuzzify_methods:
            print(f"Warning: Overwriting defuzzify method '{name}' for {cls.__name__}")
        cls._defuzzify_methods[name] = func


@total_ordering
class GFN:
    """
    Implementation of a Type-1 Gaussian Fuzzy Number (defined by mean and
    standard deviation) (m, sigma), defined by a mean (m) and a standard
    deviation (sigma). Fully implements the NumericType protocol.

    .. note::
        **Academic Note:** This class represents a standard Type-1 Gaussian
        Fuzzy Number, defined by its mean (center) and standard deviation (spread).
        The arithmetic operations implemented are common approximations that preserve
        the Gaussian form. This is distinct from the more complex Interval Type-2
        Fuzzy Sets discussed in some literature (e.g., Liu et al., 2020, Sec 6.2),
        which model uncertainty about the membership function itself.

        The arithmetic operations (`+`, `*`, etc.) for GFNs
        implemented here are common and practical approximations. True operations
        on Gaussian fuzzy numbers would result in non-Gaussian fuzzy numbers,
        requiring more complex representations. These approximations are widely
        used to maintain the Gaussian form throughout calculations but may not
        be perfectly accurate for large sigma values.
    """
    _defuzzify_methods: Dict[str, Callable] = {}

    def __init__(self, m: float, sigma: float):
        if sigma < 0:
            raise ValueError("Standard deviation (sigma) cannot be negative.")
        self.m, self.sigma = float(m), float(sigma)

    def __repr__(self) -> str:
        return f"GFN(m={self.m:.4f}, σ={self.sigma:.4f})"

    def _get_other_as_gfn(self, other: Union[GFN, Crisp, float]) -> GFN:
        if isinstance(other, GFN): return other
        val = other.value if hasattr(other, 'value') else float(other)
        return GFN(val, 0.0)

    def __add__(self, other: Union[GFN, Crisp, float]) -> GFN:
        o = self._get_other_as_gfn(other)
        # Add means, add variances (sigma^2)
        new_m = self.m + o.m
        new_sigma = np.sqrt(self.sigma**2 + o.sigma**2)
        return GFN(new_m, new_sigma)

    def __radd__(self, other: float) -> GFN:
        return self.__add__(other)

    def __sub__(self, other: Union[GFN, Crisp, float]) -> GFN:
        o = self._get_other_as_gfn(other)
        # Subtract means, add variances
        new_m = self.m - o.m
        new_sigma = np.sqrt(self.sigma**2 + o.sigma**2)
        return GFN(new_m, new_sigma)

    def __rsub__(self, other: float) -> GFN:
        o = self._get_other_as_gfn(other)
        return o - self

    def __mul__(self, other: Union[GFN, Crisp, float]) -> GFN:
        o = self._get_other_as_gfn(other)
        # Approximate multiplication
        new_m = self.m * o.m
        new_sigma = np.sqrt((self.m**2 * o.sigma**2) + (o.m**2 * self.sigma**2))
        return GFN(new_m, new_sigma)

    def __rmul__(self, other: float) -> GFN:
        return self.__mul__(other)

    def __truediv__(self, other: Union[GFN, Crisp, float]) -> GFN:
        # Division is very complex; this is a simplified approximation.
        o = self._get_other_as_gfn(other)
        if o.m == 0:
            raise ZeroDivisionError("Approximate division by a GFN with mean zero is unstable.")
        new_m = self.m / o.m
        new_sigma = np.sqrt((self.m**2 * o.sigma**2) + (o.m**2 * self.sigma**2)) / (o.m**2)
        return GFN(new_m, new_sigma)

    def __rtruediv__(self, other: float) -> GFN:
        o = self._get_other_as_gfn(other)
        return o / self

    def __pow__(self, exponent: float) -> GFN:
        # This is a simplification; true power of a GFN is complex.
        return GFN(self.m ** exponent, self.sigma * exponent)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GFN):
            return self.m == other.m and self.sigma == other.sigma
        return False

    def __lt__(self, other: Union[GFN, Crisp, float]) -> bool:
        other_mean = other.m if isinstance(other, GFN) else self._get_other_as_gfn(other).m
        return self.m < other_mean

    def inverse(self) -> GFN:
        # Approximate inverse: 1 / GFN
        if self.m == 0: raise ValueError("Cannot invert a GFN with mean zero.")
        new_m = 1.0 / self.m
        new_sigma = self.sigma / (self.m**2)
        return GFN(new_m, new_sigma)

    @staticmethod
    def neutral_element() -> GFN:
        return GFN(0.0, 0.0)

    @staticmethod
    def multiplicative_identity() -> GFN:
        return GFN(1.0, 0.0)

    @staticmethod
    def from_crisp(value: float) -> GFN:
        return GFN(m=value, sigma=0.0)

    def power(self, exponent: float) -> GFN:
        return self.__pow__(exponent)

    def alpha_cut(self, alpha: float): return NotImplemented

    def defuzzify(self, method: str = 'centroid', **kwargs) -> float:
        """
        Defuzzifies the GFN into a crisp value using various registered methods.

        Args:
            method: 'centroid'.
        """
        func = self.__class__._defuzzify_methods.get(method)
        if func is None:
            available = list(self.__class__._defuzzify_methods.keys())
            raise ValueError(f"Method '{method}' not implemented for GFN. Available: {available}")
        return func(self, **kwargs)

    @classmethod
    def get_available_defuzzify_methods(cls) -> List[str]:
        return list(cls._defuzzify_methods.keys())

    @classmethod
    def register_defuzzify_method(cls, name: str, func: Callable):
        """Registers a new defuzzification function for this number type."""
        if name in cls._defuzzify_methods:
            print(f"Warning: Overwriting defuzzify method '{name}' for {cls.__name__}")
        cls._defuzzify_methods[name] = func


@total_ordering
class IT2TrFN:
    """
    Implementation of an Interval Type-2 Trapezoidal Fuzzy Number.
    It is defined by an Upper Membership Function (UMF) and a Lower
    Membership Function (LMF), both of which are Type-1 Trapezoidal Fuzzy Numbers.

    .. note::
        **Academic Note:** Interval Type-2 Fuzzy Sets (IT2FS) capture a higher
        degree of uncertainty by modeling the membership grade itself as an
        interval. This is useful when experts are unsure even about the shape
        of the fuzzy number representing their judgment.
    """
    _defuzzify_methods: Dict[str, Callable] = {}

    def __init__(self, umf: TrFN, lmf: TrFN):
        # Validate that LMF is "inside" UMF
        if not (umf.a <= lmf.a and umf.b <= lmf.b and lmf.c <= umf.c and lmf.d <= umf.d):
            raise ValueError('LMF must be contained within the UMF')
        self.umf = umf # Upper Trapezoid
        self.lmf = lmf # Lower Trapezoid

    def __repr__(self) -> str:
        return f"IT2TrFN(UMF={self.umf}, LMF={self.lmf})"

    # --- Arithmetic Operations (Applied to both UMF and LMF) ---
    def __add__(self, other: IT2TrFN) -> IT2TrFN:
        if not isinstance(other, IT2TrFN): return NotImplemented
        return IT2TrFN(
            umf = self.umf + other.umf,
            lmf = self.lmf + other.lmf
        )

    def __mul__(self, other: IT2TrFN) -> IT2TrFN:
        if not isinstance(other, IT2TrFN): return NotImplemented
        return IT2TrFN(
            umf = self.umf * other.umf,
            lmf = self.lmf * other.lmf
        )

    def power(self, exponent: float) -> IT2TrFN:
        if not isinstance(exponent, (int, float)): return NotImplemented
        return IT2TrFN(
            umf = self.umf ** exponent,
            lmf = self.lmf ** exponent
        )

    def __pow__(self, exponent: float) -> IT2TrFN:
        return self.power(exponent)

    # Reflected and other operators
    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)
    # Subtraction and Division are complex for IT2FS and are omitted.
    def __sub__(self, other): return NotImplemented
    def __rsub__(self, other): return NotImplemented
    def __truediv__(self, other): return NotImplemented
    def __rtruediv__(self, other): return NotImplemented

    # --- Comparison (Based on the average of the defuzzified UMF and LMF) ---
    def __eq__(self, other: object) -> bool:
        if isinstance(other, IT2TrFN):
            return self.umf == other.umf and self.lmf == other.lmf
        return False

    def __lt__(self, other: IT2TrFN) -> bool:
        if not isinstance(other, IT2TrFN): return NotImplemented
        return self.defuzzify() < other.defuzzify()

    # --- Protocol Utility Methods ---
    def inverse(self) -> IT2TrFN:
        return IT2TrFN(umf=self.umf.inverse(), lmf=self.lmf.inverse())

    @staticmethod
    def neutral_element() -> IT2TrFN:
        return IT2TrFN(umf=TrFN.neutral_element(), lmf=TrFN.neutral_element())

    @staticmethod
    def multiplicative_identity() -> IT2TrFN:
        return IT2TrFN(umf=TrFN.multiplicative_identity(), lmf=TrFN.multiplicative_identity())

    @staticmethod
    def from_crisp(value: float) -> IT2TrFN:
        """Creates a crisp IT2TrFN where UMF and LMF are identical crisp TrFNs."""
        crisp_trfn = TrFN.from_crisp(value)
        return IT2TrFN(umf=crisp_trfn, lmf=crisp_trfn)

    def alpha_cut(self, alpha: float): return NotImplemented

    def defuzzify(self, method: str = 'centroid_average', **kwargs) -> float:
        """
        Defuzzifies the IT2TrFN. The standard approach is to average the
        centroids of the upper and lower membership functions.

        Args:
            method: 'centroid_average', 'centroid' (same).
        """
        func = self.__class__._defuzzify_methods.get(method)
        if func is None:
            available = list(self.__class__._defuzzify_methods.keys())
            raise ValueError(f"Method '{method}' not implemented for IT2TrFN. Available: {available}")
        return func(self, **kwargs)

    @classmethod
    def get_available_defuzzify_methods(cls) -> List[str]:
        return list(cls._defuzzify_methods.keys())

    @classmethod
    def register_defuzzify_method(cls, name: str, func: Callable):
        """Registers a new defuzzification function for this number type."""
        if name in cls._defuzzify_methods:
            print(f"Warning: Overwriting defuzzify method '{name}' for {cls.__name__}")
        cls._defuzzify_methods[name] = func


# Picture Fuzzy Sets

# ==============================================================================
# 3. GENERIC TYPE VARIABLE
# ==============================================================================

Number = TypeVar('Number', bound=NumericType)
