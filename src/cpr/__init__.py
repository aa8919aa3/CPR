"""
CPR - Current-Phase Relation
Josephson Junction Analysis Suite

A high-performance analysis suite for Josephson junction current-phase relation analysis
with advanced physics modeling, parallel processing, and publication-quality visualization.
"""

__version__ = "1.0.0"
__author__ = "CPR Team"
__email__ = "support@cpr.dev"

from .config import config
from .main_processor import EnhancedJosephsonProcessor
from .josephson_model import JosephsonFitter
from .visualization import PublicationPlotter
from .analysis_utils import PhaseAnalyzer

__all__ = [
    "config",
    "EnhancedJosephsonProcessor",
    "JosephsonFitter",
    "PublicationPlotter",
    "PhaseAnalyzer"
]
