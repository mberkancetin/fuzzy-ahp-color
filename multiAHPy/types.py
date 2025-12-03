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
    @staticmethod
    def from_saaty(value: float) -> 'NumericType': ...
    @staticmethod
    def from_normalized(value: float) -> 'NumericType': ...
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
            self.value = value.valuez
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

    @staticmethod
    def from_saaty(value: float) -> Crisp:
        return Crisp(value)

    @staticmethod
    def from_normalized(value: float) -> Crisp:
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

    @staticmethod
    def from_saaty(value: float, scale: str = "linear") -> TFN:
        """Uses FuzzyScale to create a TFN with spread (e.g., TFN(2,3,4))."""
        # We need to import FuzzyScale locally to avoid circular imports
        from .matrix_builder import FuzzyScale
        # We call the main converter, which handles scales and reciprocals
        return FuzzyScale.get_fuzzy_number(value, TFN, scale=scale) # Use a default scale

    @staticmethod
    def from_normalized(value: float) -> TFN:
        """Creates a 'crisp' TFN (e.g., TFN(0.8, 0.8, 0.8)) from a normalized value."""
        return TFN(value, value, value)

    def power(self, exponent: float) -> TFN:
        return self.__pow__(exponent)

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

    def possibility_degree(self, other: TFN, tolerance: float | None = None) -> float:
        """
        Calculate the possibility degree that self >= other.
        V(self >= other), used in Chang's extent analysis method.
        """
        from .config import configure_parameters
        final_tolerance = tolerance if tolerance is not None else configure_parameters.FLOAT_TOLERANCE

        if self.m >= other.m:
            return 1.0
        if other.l >= self.u:
            return 0.0
        denominator = (self.m - self.u) - (other.m - other.l)
        if abs(denominator) < final_tolerance:
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
    def from_saaty(value: float) -> TrFN:
        from .matrix_builder import FuzzyScale
        return FuzzyScale.get_fuzzy_number(value, TrFN, scale='linear')

    @staticmethod
    def from_normalized(value: float) -> TrFN:
        return TrFN(value, value, value, value)

    @staticmethod
    def from_tfn(tfn: TFN) -> TrFN:
        """Converts a TFN to a degenerate TrFN."""
        return TrFN(tfn.l, tfn.m, tfn.m, tfn.u)

    def power(self, exponent: float) -> TrFN:
        return self.__pow__(exponent)

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        if not (0 <= alpha <= 1): raise ValueError("Alpha must be between 0 and 1.")
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

    def __init__(self, mu: float, nu: float, ifn_scale_name: str = None):
        from .config import configure_parameters
        self.mu = float(mu)
        self.nu = float(nu)
        if self.mu + self.nu > 1.0 + 1e-9:
            raise ValueError(f"Invalid IFN: mu({mu}) + nu({nu}) > 1")
        self.pi = 1.0 - self.mu - self.nu

        self.pi = 1.0 - self.mu - self.nu
        self.ifn_scale_name = ifn_scale_name
        self.ifn_scale = None
        if self.ifn_scale_name is None:
            self.ifn_scale = configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS["buyukozkan_9_level"]
        else:
            self.ifn_scale = configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS[self.ifn_scale_name]

    def __repr__(self) -> str:
        return f"IFN(μ={self.mu:.4f}, ν={self.nu:.4f})"

    def _get_other_as_ifn(self, other: Union[IFN, Crisp, float]) -> IFN:
        """Helper to convert other types to IFN for operations."""
        if isinstance(other, IFN): return other
        # A crisp number has no hesitation, so
        # ν = 1 - μ
        if not isinstance(other, IFN):
            key = other.value if hasattr(other, 'value') else float(other)
            val = self.ifn_scale[key]
            return IFN(mu=val[0], nu=1-val[1])
        else:
            return IFN(mu=other.mu, nu=other.nu)

    def __add__(self, other: IFN) -> IFN:
        if not isinstance(other, IFN): return NotImplemented
        return IFN(
            self.mu + other.mu - self.mu * other.mu,
            self.nu * other.nu
            )

    def __mul__(self, other: IFN) -> IFN:
        if not isinstance(other, IFN): return NotImplemented
        return IFN(
            self.mu * other.mu,
            self.nu + other.nu - self.nu * other.nu
            )

    # Subtraction and Division are not standardly defined for IFS in a way that
    # is useful for AHP. They are often context-specific or not used at all.
    # Returning NotImplemented is the correct, safe approach.
    def __sub__(self, other): return NotImplemented

    def __truediv__(self, scalar: float) -> IFN:
        """
        Performs scalar division on an IFN (A / λ), where λ is a crisp number.
        This is a crucial operation for normalizing a set of fuzzy weights.

        The formula ensures that the 'score' of the resulting IFN is scaled
        proportionally, while preserving the core structure.
        """
        if not isinstance(scalar, (int, float, np.number)):
            return NotImplemented
        if scalar <= 0:
            raise ZeroDivisionError("Cannot divide an IFN by a non-positive scalar.")

        # This normalization formula is derived from several sources on IFN arithmetic
        # to ensure that the score function behaves somewhat linearly for λ > 1.
        # Score = μ + α*π. We want Score' ≈ Score / λ

        # A simple and effective scaling is to scale the membership and hesitation.
        # This is a robust heuristic.

        original_score = self.defuzzify(method='centroid') # Get score with hesitation
        target_score = original_score / scalar

        # Use our robust constructor to build the new, scaled IFN
        # We scale the hesitation by the same factor.
        scaled_hesitation = self.pi / scalar

        return IFN.from_score_and_hesitation(target_score, scaled_hesitation)

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
            self.mu ** exponent,
            1 - (1 - self.nu) ** exponent
            )

    def __pow__(self, exponent: float) -> IFN:
        return self.power(exponent)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IFN): return NotImplemented
        return np.isclose(self.mu, other.mu) and np.isclose(self.nu, other.nu)

    def __lt__(self, other: IFN) -> bool:
        if not isinstance(other, IFN): return NotImplemented

        # 1. Compare Scores (S = mu - nu)
        s1 = self.defuzzify(method='score')
        s2 = other.defuzzify(method='score')

        if not np.isclose(s1, s2):
            return s1 < s2

        # 2. Tie-breaker: Compare Accuracy (H = mu + nu)
        # Higher accuracy is "better" (greater), so lower accuracy is "less than"
        a1 = self.defuzzify(method='accuracy')
        a2 = other.defuzzify(method='accuracy')

        return a1 < a2

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
    def from_crisp_retired(value: float) -> IFN:
        """Creates an IFN from a crisp value [0,1], assuming no hesitation."""
        if not (0 <= value <= 1):
            raise ValueError("Crisp value for IFN conversion must be between 0 and 1.")
        return IFN(mu=value, nu=1.0 - value)

    @staticmethod
    def from_saaty_experimental(value: float) -> IFN:
        """
        Creates an IFN from a crisp Saaty-scale value (e.g., 1-9 and reciprocals).

        This method uses a precise interpolation function for non-integer values
        and correctly handles reciprocals.

        Args:
            value: The Saaty-scale judgment (e.g., 3.5, 1/5).
            interpolation_func: A function that takes a float and returns a
                                tuple (mu, nu). If None, a simple rounding
                                method is used as a fallback.
        """
        from .config import configure_parameters
        interpolation_func = configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS.get("buyukozkan_9_level")

        if value <= 0:
            raise ValueError("Saaty-scale value must be positive.")

        # Handle the base case of "equal importance"
        if np.isclose(value, 1.0):
            # The Nguyen scale for 1 is (0.50, 0.40)
            from .config import configure_parameters # Local import for safety
            return IFN(*configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS['buyukozkan_9_level'](1.0))

        is_reciprocal = False
        if value < 1:
            is_reciprocal = True
            value = 1.0 / value

        # Use the provided interpolation function for precision
        if interpolation_func:
            mu, nu = interpolation_func(value)
        else:
            # Fallback if no function is provided: round to nearest integer key
            from .config import configure_parameters
            key = int(round(np.clip(value, 1, 9)))
            mu, nu = configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS['buyukozkan_9_level'](float(key))

        if is_reciprocal:
            return IFN(mu=nu, nu=mu)  # Return the inverse
        else:
            return IFN(mu=mu, nu=nu)

    @staticmethod
    def from_saaty_with_consistency(
        value: float,
        matrix_cr: float,
        hesitation_factor: float = 0.2, # Your requested 0.2 factor
        base_hesitation: float = 0.05   # Your requested 0.05 base
    ) -> IFN:
        """
        Creates an IFN where hesitation is derived from the Matrix CR.
        Even for value=1.0 (Diagonal), hesitation is applied.

        Formula: pi = max(base, factor * min(CR, 1.0))
        CRITICAL: Applies hesitation logic even to the Diagonal (value=1.0).
        """
        if value <= 0: raise ValueError("Saaty value must be > 0")

        # 1. Calculate Dynamic Hesitation (Pi)
        # Pi increases as CR increases.
        # Example: if CR=0.1, factor=0.2, base=0.05 -> pi = 0.07
        cr_capped = min(abs(matrix_cr), 1.0)
        pi = base_hesitation + (hesitation_factor * cr_capped)

        # Cap pi to ensure we don't break math (must be < 1)
        pi = min(pi, 0.99)

        # 2. Determine Linguistic Anchor (The "Core" Meaning)
        if np.isclose(value, 1.0):
            # For Diagonal/Equal: The core meaning is exactly 50/50.
            # Anchor represents the split between Mu and Nu if Pi were 0.
            anchor = 0.5
        else:
            # For other values, use Logarithmic Scale to find position in [0,1]
            scale_base = 9
            # Handle reciprocals naturally via log (log(1/x) = -log(x))
            log_val = np.log(value) / np.log(scale_base)
            # Map [-1, 1] to [0, 1]
            anchor = (log_val + 1) / 2.0
            anchor = np.clip(anchor, 0.01, 0.99)

        # 3. Distribute Mass based on Hesitation
        # The available mass for Membership + Non-Membership is (1 - pi).
        # We distribute this mass according to the Anchor ratio.

        mu = (1 - pi) * anchor
        nu = (1 - pi) * (1 - anchor)

        return IFN(mu, nu)


    @staticmethod
    def from_saaty(value: float) -> IFN:
        """
        Converts a Saaty scale value to an IFN using academically cited scales.
        """
        from .config import configure_parameters
        scale = configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS["buyukozkan_9_level"]

        # 1. Handle Reciprocals (Values < 1)
        if value < 1.0 - 1e-9:
            # The reciprocal of an IFN A is (nu, mu)
            # We convert the whole number part (1/val) and then invert it
            return IFN.from_saaty(round((1.0 / value), 1)).inverse()

        # 2. Handle Exact Integers (Direct Lookup)
        # Round to nearest to catch floating point noise (e.g. 3.0000001)
        val_round = round(value, 2)

        if val_round == int(val_round):
            mu, nu = scale(int(val_round))
            return IFN(mu, nu)

        # 3. Handle Aggregated/Continuous Values (Interpolation)
        # If we have a value like 3.5 (from a geometric mean), we interpolate
        # between the defined scale points. This is an engineering necessity.

        # Clamp to 1-9 range
        v = np.clip(value, 1, 9)
        lower = int(np.floor(v))
        upper = int(np.ceil(v))

        if lower == upper:
            mu, nu = scale(lower)
            return IFN(mu, nu)

        # Linear Interpolation between the academic points
        ratio = v - lower
        mu_l, nu_l = scale(lower)
        mu_u, nu_u = scale(upper)

        mu = mu_l + ratio * (mu_u - mu_l)
        nu = nu_l + ratio * (nu_u - nu_l)

        return IFN(mu, nu)

    @staticmethod
    def from_normalized(value: float) -> IFN:
        """
        Creates an IFN from a normalized value (e.g., weight or performance score)
        in the [0, 1] range, assuming no hesitation.
        """
        return IFN(mu=value, nu=1.0 - value)

    @staticmethod
    def from_crisp(value: float) -> IFN:
        """
        Default crisp conversion method. Assumes the value is from the Saaty scale.
        For converting normalized [0,1] values, use `from_normalized()` instead.
        """
        return IFN.from_saaty(value)

    @staticmethod
    def from_score_and_hesitation(score: float, hesitation: float, alpha: float = 0.5) -> IFN:
        """
        Constructs an IFN from a target defuzzified score and a hesitation value.
        This is the reverse of the Xu-Yager defuzzification.

        score = μ + α * π
        μ + ν + π = 1
        """
        score = np.clip(score, 0, 1)
        pi = np.clip(hesitation, 0, 1)

        # from score = μ + α*π  =>  μ = score - α*π
        mu = score - alpha * pi

        # nu = 1 - μ - π
        nu = 1 - mu - pi

        # Final check and correction for validity.
        # If this process made `mu` or `nu` invalid (e.g., negative), it means the
        # hesitation is too large for the score. We must adjust.
        if mu < 0 or nu < 0 or mu + nu > 1:
            # If `mu` went negative, it means `alpha * pi > score`.
            # In this case, the score is entirely composed of hesitation.
            mu = 0
            # The new hesitation must be such that `alpha * pi_new = score`
            pi_new = score / alpha if alpha > 0 else 0
            pi_new = min(pi_new, 1.0) # Hesitation cannot be > 1

            nu = 1 - pi_new
            pi = pi_new # Update pi for the final object

        # Create the IFN. Our constructor has built-in validation.
        return IFN(mu, nu)

    def alpha_cut(self, alpha: float): return NotImplemented

    def defuzzify(self, method: str = 'centroid', **kwargs) -> float:
        """
        Defuzzifies the IFN into a crisp value using various registered methods.

        Args:
            method: 'centroid', 'score', 'entropy', 'accuracy', 'value'.
        """
        func = self.__class__._defuzzify_methods.get(method)
        if func: return func(self, **kwargs)
        raise ValueError(f"Method {method} not found")

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
        # This is a simplified approximation.
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
        # This is a simplification.
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
        if not (umf.a <= lmf.a and umf.b <= lmf.b and lmf.c <= umf.c and lmf.d <= umf.d):
            raise ValueError('LMF must be contained within the UMF')
        self.umf = umf # Upper Trapezoid
        self.lmf = lmf # Lower Trapezoid

    def __repr__(self) -> str:
        return f"IT2TrFN(UMF={self.umf}, LMF={self.lmf})"

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

    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __sub__(self, other): return NotImplemented
    def __rsub__(self, other): return NotImplemented
    def __truediv__(self, other): return NotImplemented
    def __rtruediv__(self, other): return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IT2TrFN):
            return self.umf == other.umf and self.lmf == other.lmf
        return False

    def __lt__(self, other: IT2TrFN) -> bool:
        if not isinstance(other, IT2TrFN): return NotImplemented
        return self.defuzzify() < other.defuzzify()

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


# ==============================================================================
# 3. GENERIC TYPE VARIABLE
# ==============================================================================

Number = TypeVar('Number', bound=NumericType)
