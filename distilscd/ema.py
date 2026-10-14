"""Moyenne mobile exponentielle des poids (EMA).

Un second jeu de poids, mis à jour après chaque pas d'optimiseur par
`ema <- d·ema + (1-d)·model`, sert à la validation et au checkpoint. Il ne coûte
rien à l'entraînement (pas de gradient) et rien à l'inférence (mêmes tenseurs,
même modèle) : il change seulement *quelles* valeurs on garde.

Deux raisons de l'essayer ici en particulier. La recette retenue tourne à **LR
constant** sur 200 époques — donc sans la décroissance qui, dans un cosine,
moyenne implicitement les dernières époques ; les courbes de fin sont bruitées,
c'est noté au journal depuis août. Et `best.pt` retient le **maximum** d'une
courbe bruitée, ce qui gonfle mécaniquement le chiffre rapporté : lisser les
poids attaque le bruit à sa source plutôt qu'après coup.

⚠️ **L'état EMA doit être sauvegardé dans `last.pt`.** Un run à 200 époques dure
~15 h 30 et se soumet à `--time=20:00:00` : il tient normalement en une fois. Mais
« normalement » ne suffit pas — préemption, dépassement, nœud qui tombe, ou
simple relance à 300 époques : dès qu'une reprise a lieu, un EMA non sauvegardé
repartirait des poids courants, effaçant sa moyenne. Sans message d'erreur, et
avec pour seul symptôme un résultat un peu moins bon qu'il n'aurait dû être.
Exactement le mode de défaillance silencieux qui a déjà coûté trois fois à ce
projet (paramètre accepté mais non propagé).
"""

from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup: int = 2000):
        self.decay = decay
        self.warmup = warmup
        self.updates = 0
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _current_decay(self) -> float:
        """Décote au démarrage : sinon l'EMA reste collée à l'initialisation.

        `(1+t)/(10+t)` vaut 0,1 au premier pas et rejoint `decay` en quelques
        milliers d'itérations — la moyenne suit d'abord le modèle, puis se fige.
        """
        if self.warmup <= 0:
            return self.decay
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self._current_decay()
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1.0 - d)
            else:
                # Compteurs entiers (num_batches_tracked...) : une moyenne n'a pas
                # de sens, on recopie.
                v.copy_(src[k])

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates,
                "decay": self.decay, "warmup": self.warmup}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = state["updates"]
        self.decay = state["decay"]
        self.warmup = state["warmup"]
