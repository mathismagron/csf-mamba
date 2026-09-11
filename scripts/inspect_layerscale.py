"""Le bloc d'attention s'est-il seulement allumé ?

⚗️ Outil de la piste expérimentale (`documentation/hybride.md`).

Chaque résidu du bloc Transformer est multiplié par un `gamma` appris, initialisé
à 1e-5 pour que le bloc soit l'identité au départ. Toute la question est de savoir
ce que l'entraînement en a fait :

* **gamma resté proche de 1e-5** — le bloc n'a jamais contribué. Le résultat
  négatif ne dit alors RIEN de l'attention : il dit que le bloc est resté éteint,
  et il faut chercher pourquoi (initialisation trop timide, weight decay qui le
  ramène à zéro, taux d'apprentissage inadapté).
* **gamma de l'ordre de 1e-2 ou plus** — le bloc a bel et bien été utilisé, et il
  a quand même dégradé le modèle. Le résultat négatif porte alors sur l'idée
  elle-même, et régler l'optimisation n'y changera rien.

Distinguer les deux coûte deux minutes et décide s'il vaut la peine de relancer
huit entraînements.

    python -m scripts.inspect_layerscale $SCRATCH/csf-mamba-runs/second_mini_chess_hyb-*/best.pt
"""

import argparse
import statistics
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    args = p.parse_args()

    print(f"{'run':<34} {'paramètre':<26} {'|gamma| moyen':>14} {'max':>10}  état")
    print("-" * 96)
    for chemin in args.checkpoints:
        etat = torch.load(chemin, map_location="cpu", weights_only=False)
        if isinstance(etat, dict) and "model" in etat:
            etat = etat["model"]
        nom = Path(chemin).parent.name.replace("second_mini_chess_", "")
        gammas = {k: v for k, v in etat.items() if k.endswith(("gamma1", "gamma2"))}
        if not gammas:
            print(f"{nom:<34} aucun gamma — ce checkpoint n'est pas un modèle hybride")
            continue
        moyennes = []
        for k, v in sorted(gammas.items()):
            m, mx = v.abs().mean().item(), v.abs().max().item()
            moyennes.append(m)
            # 1e-5 est la valeur d'initialisation : au-delà d'un facteur 100, le
            # bloc a réellement appris à contribuer.
            etat_txt = ("ÉTEINT" if m < 1e-4 else
                        "faible" if m < 1e-3 else
                        "actif" if m < 1e-2 else "TRÈS actif")
            court = k.replace("attn.", "").replace(".blocks.", ".b")
            print(f"{nom:<34} {court:<26} {m:>14.2e} {mx:>10.2e}  {etat_txt}")
        g = statistics.fmean(moyennes)
        print(f"{'':<34} {'-> moyenne du run':<26} {g:>14.2e} {'':>10}  "
              f"{'facteur ' + format(g / 1e-5, '.0f') + ' x l initialisation'}")
        print()


if __name__ == "__main__":
    main()
