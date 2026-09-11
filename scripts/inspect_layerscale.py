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

⚠️ Le venv doit être activé, même sur un nœud de connexion — le script lit un
checkpoint PyTorch :

    module load python/3.11 cuda/12.2
    source $SCRATCH/csf-venv-cu12/bin/activate
    python -m scripts.inspect_layerscale \\
        $SCRATCH/csf-mamba-runs/second_mini_chess_hyb-eff-s3-d2-s1/best.pt \\
        $SCRATCH/csf-mamba-runs/second_mini_chess_hyb-perf-s3-d2-s1/best.pt
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
    resume = []
    for chemin in args.checkpoints:
        # `best.pt` est un state_dict nu : weights_only=True suffit, évite
        # l'avertissement de dépickling et va plus vite.
        try:
            etat = torch.load(chemin, map_location="cpu", weights_only=True)
        except Exception:
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
        resume.append((nom, g))
        print(f"{'':<34} {'-> moyenne du run':<26} {g:>14.2e} {'':>10}  "
              f"facteur {g / 1e-5:.0f} x l'initialisation")
        print()

    if not resume:
        return
    pire = min(g for _, g in resume)
    print("=" * 96)
    if pire < 1e-4:
        print("VERDICT : le bloc d'attention est resté ÉTEINT.")
        print("  Le résultat négatif ne porte pas sur l'attention mais sur le fait")
        print("  qu'elle n'a jamais contribué. Régler l'optimisation est indispensable")
        print("  avant de conclure : --attn-no-wd et --attn-layer-scale 1e-1.")
    elif pire < 1e-2:
        print("VERDICT : le bloc s'est allumé FAIBLEMENT.")
        print("  Contribution réelle mais ténue. Relancer avec --attn-no-wd et")
        print("  --attn-layer-scale 1e-1 vaut la peine, sans garantie.")
    else:
        print("VERDICT : le bloc a bien été UTILISÉ, et il a quand même dégradé le")
        print("  modèle. Le résultat négatif porte sur l'idée elle-même ; régler")
        print("  l'optimisation n'y changera pas grand-chose. Seul le dropout garde")
        print("  un sens, contre l'hypothèse de surapprentissage.")


if __name__ == "__main__":
    main()
