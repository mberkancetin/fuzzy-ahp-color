from __future__ import annotations
import numpy as np
from typing import Protocol, TypeVar, Union, Any
from functools import total_ordering
from multiAHPy.defuzzification import Defuzzification

# ==============================================================================
# 1. THE PROTOCOL BLUEPRINT
# ==============================================================================

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
    def defuzzify(self, method: str = 'centroid', **kwargs) -> float: ...
    def power(self, exponent: float) -> 'NumericType': ...


# ==============================================================================
# 2. IMPLEMENTATIONS OF THE PROTOCOL
# ==============================================================================

@total_ordering
class Crisp:
    """
    A wrapper for float to make it conform to the NumericType protocol.
    """

    def __init__(self, value: float):
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

    def defuzzify(self, method: str = 'centroid', **kwargs) -> float:
        return Defuzzification.defuzzify(self, method, **kwargs)

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


@total_ordering
class TFN:
    """
    Triangular Fuzzy Number (TFN) class.
    A TFN is represented as (l, m, u) where l ≤ m ≤ u.
    l: lower bound, m: middle value, u: upper bound
    """

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
        o = self._get_other_as_tfn(other)
        return self * o.inverse()

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

    # Additional methods as needed
    def defuzzify(self, method: str = 'centroid', **kwargs) -> float:
        """
        Delegates defuzzification to the external Defuzzification class.
        This keeps the data object (TFN) separate from the algorithm.
        """
        return Defuzzification.defuzzify(self, method, **kwargs)

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


@total_ordering
class TrFN:
    """
    Implementation of a Trapezoidal Fuzzy Number (a, b, c, d).
    """

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
        o = self._get_other_as_trfn(other)
        return self * o.inverse()

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

    def defuzzify(self, method: str = 'centroid', **kwargs) -> float:
        """Delegates defuzzification to the external Defuzzification class."""
        return Defuzzification.defuzzify(self, method, **kwargs)

    def power(self, exponent: float) -> TrFN:
        return self.__pow__(exponent)


@total_ordering
class GFN:
    """
    Implementation of a Gaussian Fuzzy Number, defined by a mean (m) and a
    standard deviation (sigma). Fully implements the NumericType protocol.
    """
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

    def defuzzify(self, method: str = 'centroid', **kwargs) -> float:
        return Defuzzification.defuzzify(self, method, **kwargs)


# ==============================================================================
# 3. GENERIC TYPE VARIABLE
# ==============================================================================

Number = TypeVar('Number', bound=NumericType)
