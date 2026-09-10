"""Pistes expérimentales, tenues à l'écart du modèle de référence.

Ce sous-paquet contient des composants **hors du cadre initial du stage**. Rien
de ce qu'il contient n'est actif par défaut : le modèle de référence construit
exactement les mêmes tenseurs qu'avant son ajout, et un test le vérifie.

Règle de séparation retenue : le code vit ici, se lance par un sbatch dédié
(`scripts/train_hybrid.sbatch`), tague ses runs `hyb-*`, et se documente dans
`documentation/hybride.md` — pas dans le journal de bord de la campagne.
"""

from .bitemporal_attention import BiTemporalAttention

__all__ = ["BiTemporalAttention"]
