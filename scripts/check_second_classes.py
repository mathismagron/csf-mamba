"""Vérifie l'association indice -> classe sémantique de SECOND.

Pourquoi : nos `CLASS_NAMES` (`csf_mamba/datasets/second.py`) suivent l'ordre de
l'article SECOND, mais le code de ChangeMamba utilise un ordre différent pour les
indices 1, 2 et 4 (eau / sol / arbre permutés). Les deux ne peuvent pas être
justes. Aucune métrique n'en dépend — tout est calculé sur des indices, de façon
cohérente — mais un nom de classe faux dans le rapport serait une erreur factuelle.

Comment : les dossiers `GT_T*_COLORED`, ignorés par le dataloader, sont les mêmes
cartes en RGB. Pour chaque indice on relève la couleur dominante des pixels
correspondants, puis on la compare à la palette officielle. Le lien couleur -> nom
est lui non ambigu (bleu = eau, vert = végétation), c'est donc lui qui sert
d'ancrage.

    python -m scripts.check_second_classes --data-root $SCRATCH/SECOND

Ne demande ni GPU ni environnement particulier : lecture de PNG uniquement.
"""

import argparse
import ast
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

_T0 = time.monotonic()


def log(msg: str) -> None:
    """Trace horodatée, pour situer un blocage plutôt que le deviner."""
    print(f"[{time.monotonic() - _T0:7.1f} s] {msg}", flush=True)


def read_class_names() -> tuple[str, ...]:
    """Lit CLASS_NAMES dans le source, SANS importer csf_mamba.

    `csf_mamba.datasets.second` importe torch, dont le chargement depuis le venv
    sur Lustre est ce qui a fait échouer le premier job : 20 minutes sans même
    atteindre le premier print. Ce script n'a besoin que d'un tuple de chaînes —
    on l'extrait par analyse syntaxique. Pas de duplication, pas de torch, et le
    script ne dépend plus que de numpy et Pillow.
    """
    src = Path(__file__).resolve().parents[1] / "csf_mamba" / "datasets" / "second.py"
    for node in ast.parse(src.read_text()).body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "CLASS_NAMES" for t in node.targets)):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit(f"CLASS_NAMES introuvable dans {src}")

# Palette officielle de SECOND, telle que reprise par ChangeMamba
# (`ST_COLORMAP` / `ST_CLASSES`). C'est l'association couleur -> nom qui fait foi.
REFERENCE_PALETTE = {
    (255, 255, 255): "inchangé / non annoté",
    (0, 0, 255): "eau",
    (128, 128, 128): "sol non végétalisé",
    (0, 128, 0): "végétation basse",
    (0, 255, 0): "arbre",
    (128, 0, 0): "bâtiment",
    (255, 0, 0): "terrain de sport",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=200,
                   help="Nombre de tuiles échantillonnées (0 = toutes).")
    p.add_argument("--date", default="T2", choices=["T1", "T2"])
    return p.parse_args()


def main():
    args = parse_args()
    log("démarrage")
    class_names = read_class_names()
    log(f"CLASS_NAMES lus : {class_names}")

    root = Path(args.data_root) / args.split
    idx_dir = root / f"GT_{args.date}"
    col_dir = root / f"GT_{args.date}_COLORED"
    for d in (idx_dir, col_dir):
        if not d.is_dir():
            raise SystemExit(f"dossier introuvable : {d}")

    names = sorted(p.name for p in idx_dir.glob("*.png"))
    log(f"listage terminé : {len(names)} fichiers")
    if args.limit:
        # Pas échantillonné en tête de liste : les tuiles y sont spatialement
        # contiguës, donc non représentatives (leçon du comptage Hi-UCD).
        names = names[:: max(1, len(names) // args.limit)][: args.limit]
    print(f"{len(names)} tuiles échantillonnées dans {idx_dir}")

    counts = {c: Counter() for c in range(7)}
    for i, name in enumerate(names):
        if i % 25 == 0:
            log(f"tuile {i}/{len(names)}")
        idx = np.asarray(Image.open(idx_dir / name))
        col = np.asarray(Image.open(col_dir / name).convert("RGB"))
        if idx.ndim != 2 or col.shape[:2] != idx.shape:
            raise SystemExit(f"formes incohérentes pour {name}")
        keys = (col[..., 0].astype(np.int64) << 16 | col[..., 1].astype(np.int64) << 8
                | col[..., 2].astype(np.int64))
        for c in range(7):
            sel = keys[idx == c]
            if sel.size:
                key, n = np.unique(sel, return_counts=True)
                counts[c].update(dict(zip(key.tolist(), n.tolist())))

    log("lecture terminée")
    print(f"\n{'idx':>3} | {'couleur dominante':>18} | {'pureté':>7} | "
          f"{'palette officielle':<22} | {'nos CLASS_NAMES':<22} | ok")
    print("-" * 100)
    verdict = True
    for c in range(7):
        if not counts[c]:
            print(f"{c:>3} | {'(absent)':>18} |")
            continue
        key, n = counts[c].most_common(1)[0]
        rgb = ((key >> 16) & 255, (key >> 8) & 255, key & 255)
        purity = n / sum(counts[c].values())
        official = REFERENCE_PALETTE.get(rgb, "⚠️ couleur hors palette")
        ours = class_names[c] if c < len(class_names) else "?"
        match = official.split(" /")[0] in _fr(ours) or c == 0
        verdict &= match
        print(f"{c:>3} | {str(rgb):>18} | {purity:6.1%} | {official:<22} | "
              f"{ours:<22} | {'✓' if match else '✗'}")

    print("\n" + ("✓ nos CLASS_NAMES sont corrects" if verdict else
                  "✗ nos CLASS_NAMES sont FAUX — corriger csf_mamba/datasets/second.py"))
    print("(la pureté doit être ~100 % : une valeur basse signalerait des cartes "
          "colorées désynchronisées des cartes d'indices)")


def _fr(name: str) -> str:
    """Traduit nos noms anglais pour la comparaison avec la palette officielle."""
    return {
        "reserved": "inchangé",
        "non-vegetated ground": "sol non végétalisé",
        "tree": "arbre",
        "low vegetation": "végétation basse",
        "water": "eau",
        "building": "bâtiment",
        "playground": "terrain de sport",
    }.get(name, name)


if __name__ == "__main__":
    main()
