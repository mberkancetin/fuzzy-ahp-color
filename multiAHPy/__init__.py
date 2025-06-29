__version__ = "0.0.7.0"

from multiAHPy import defuzzification
from .config import configure_parameters
from .model import Hierarchy, Node, Alternative
from .types import Crisp, TFN, TrFN, GFN, IFN, IT2TrFN

from .weight_derivation import register_weight_method
from .aggregation import register_aggregation_method
from .consistency import register_consistency_method
