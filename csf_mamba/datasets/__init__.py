"""Registre des datasets SCD supportés.

Chaque entrée : (classe Dataset, nombre de canaux sémantiques).
Le nombre de canaux inclut l'index 0 réservé (convention A) : Hi-UCD = 9 classes
réelles + 1, SECOND = 6 classes réelles + 1, Landsat-SCD = 4 classes réelles + 1.
"""

from .hi_ucd import NUM_SEMANTIC_CLASSES as HI_UCD_CLASSES
from .hi_ucd import HiUCDDataset
from .landsat_scd import NUM_SEMANTIC_CLASSES as LANDSAT_CLASSES
from .landsat_scd import LandsatSCDDataset
from .second import NUM_SEMANTIC_CLASSES as SECOND_CLASSES
from .second import SECONDDataset

DATASETS = {
    "hi_ucd": (HiUCDDataset, HI_UCD_CLASSES),
    "second": (SECONDDataset, SECOND_CLASSES),
    "landsat_scd": (LandsatSCDDataset, LANDSAT_CLASSES),
}

__all__ = ["DATASETS", "HiUCDDataset", "SECONDDataset", "LandsatSCDDataset"]
