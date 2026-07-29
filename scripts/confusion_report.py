"""Décompose une matrice de confusion SCD : localisation vs sémantique.

Répond à : « les erreurs viennent-elles de la DÉTECTION du changement (faux
positifs / faux négatifs) ou de sa CLASSIFICATION (bonne zone, mauvaise
classe) ? » — les deux appellent des correctifs opposés.

    python -m scripts.confusion_report <run>/eval_<split>/confusion.csv
"""

import sys

import numpy as np


def report(path):
    h = np.loadtxt(path, delimiter=",")
    fn = h[0, 1:].sum()          # prédit « pas de changement », en réalité changé
    fp = h[1:, 0].sum()          # prédit « changement », en réalité inchangé
    tp = h[1:, 1:].sum()         # changement détecté (classe correcte ou non)
    ok = np.diag(h[1:, 1:]).sum()
    sem_err = tp - ok

    print(f"\n===== {path} =====")
    print(f"  vrais négatifs (inchangé, bien vu) : {h[0, 0]:14.0f}")
    print(f"  faux négatifs  (changement manqué) : {fn:14.0f}")
    print(f"  faux positifs  (changement inventé): {fp:14.0f}")
    print(f"  changement détecté                 : {tp:14.0f}")
    print(f"     dont bien classé                : {ok:14.0f}"
          f"  ({100 * ok / max(tp, 1):.1f} %)")
    print(f"     dont mal classé                 : {sem_err:14.0f}")
    print()
    print(f"  erreurs de LOCALISATION (FP+FN)    : {fn + fp:14.0f}")
    print(f"  erreurs de SÉMANTIQUE              : {sem_err:14.0f}")
    if sem_err > 0:
        ratio = (fn + fp) / sem_err
        print(f"  ratio localisation / sémantique    : {ratio:14.2f}")
        verdict = ("LOCALISATION" if ratio > 1.5 else
                   "SÉMANTIQUE" if ratio < 0.67 else "les DEUX à parts comparables")
        print(f"  -> goulot d'étranglement : {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m scripts.confusion_report <confusion.csv> [...]")
    for p in sys.argv[1:]:
        report(p)
