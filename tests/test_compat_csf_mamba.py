"""Compatibilité ascendante du renommage csf_mamba -> distilscd.

Vérifie que la cale `csf_mamba/` (cf. documentation/rename.md) :

1. redirige `csf_mamba.X` vers le module `distilscd.X` lui-même (même objet),
   avec les alias `CSFMamba` / `CSFMambaLoss` et un DeprecationWarning ;
2. permet de **désérialiser un objet picklé sous l'ancien chemin de module**
   (`csf_mamba.model.CSFMamba`, sous-modules compris) : par `torch.load` dans un
   interpréteur neuf où ni `csf_mamba` ni `distilscd` n'ont été importés — c'est
   la situation d'un vieux checkpoint ouvert après le renommage ;
3. lit un pickle protocole 0 écrit à la main avec l'ancien nom (aucun recours au
   code renommé pour l'écrire).

    python -m pytest tests/test_compat_csf_mamba.py -v
"""

import io
import os
import pickle
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Exécute `code` dans un interpréteur neuf (sys.modules vierge)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")])
    env.setdefault("KMP_AFFINITY", "disabled")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)


def test_shim_aliases_same_modules():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import csf_mamba.model as old_model
        from csf_mamba.losses.composite import CSFMambaLoss
        from csf_mamba.model import CSFMamba
    import distilscd.losses.composite
    import distilscd.model

    assert old_model is distilscd.model
    assert sys.modules["csf_mamba.model"] is distilscd.model
    assert CSFMamba is distilscd.model.DistilSCDStudent
    assert CSFMambaLoss is distilscd.losses.composite.DistilSCDLoss
    # le module réel garde sa propre identité
    assert distilscd.model.__spec__.name == "distilscd.model"
    assert distilscd.model.__name__ == "distilscd.model"


def test_shim_emits_deprecation_warning():
    res = _run_fresh("""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import csf_mamba
        assert any(issubclass(x.category, DeprecationWarning) and "distilscd" in str(x.message)
                   for x in w), [str(x.message) for x in w]
        print("OK")
    """)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr


def _pickle_under_old_path(obj) -> bytes:
    """Sérialise `obj` (torch.save) comme l'aurait fait le code d'avant le renommage.

    Toutes les classes de `distilscd.*` reçoivent temporairement
    `__module__ = 'csf_mamba.*'`, et la classe du modèle son ancien nom
    `CSFMamba` : le flux produit référence exclusivement l'ancien chemin.
    """
    import csf_mamba  # noqa: F401  (pickle vérifie que l'ancien chemin résout)
    import distilscd.model

    patched = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("distilscd") or mod is None:
            continue
        for val in vars(mod).values():
            if isinstance(val, type) and val.__module__ == name:
                patched.append((val, val.__module__, val.__qualname__))
                val.__module__ = "csf_mamba" + name[len("distilscd"):]
    cls = distilscd.model.DistilSCDStudent
    cls.__qualname__ = "CSFMamba"
    try:
        buf = io.BytesIO()
        torch.save(obj, buf)  # même Pickler que train.py ; zip non compressé
        return buf.getvalue()
    finally:
        for val, module, qualname in patched:
            val.__module__, val.__qualname__ = module, qualname
        cls.__qualname__ = "DistilSCDStudent"


def test_unpickle_old_module_path_after_rename(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from distilscd.model import DistilSCDStudent

        torch.manual_seed(0)
        model = DistilSCDStudent(num_semantic_classes=7, encoder="conv", backend="ref").eval()
        x1, x2 = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            ref = model(x1, x2)
        blob = _pickle_under_old_path({"model_obj": model, "epoch": 3})

    assert b"csf_mamba.model" in blob and b"CSFMamba" in blob
    assert b"distilscd" not in blob, "le flux doit référencer uniquement l'ancien chemin"

    ckpt = tmp_path / "old_style.pt"
    with ckpt.open("wb") as f:
        f.write(blob)
    torch.save({"x1": x1, "x2": x2, **{f"ref_{k}": v for k, v in ref.items()}},
               tmp_path / "io.pt")

    # Interpréteur neuf : rien n'est importé, pickle doit trouver la cale seul.
    res = _run_fresh(f"""
        import sys, warnings, torch
        warnings.simplefilter("ignore", DeprecationWarning)
        assert "csf_mamba" not in sys.modules and "distilscd" not in sys.modules
        ck = torch.load({str(ckpt)!r}, map_location="cpu", weights_only=False)
        io = torch.load({str(tmp_path / "io.pt")!r}, map_location="cpu", weights_only=True)
        from distilscd.model import DistilSCDStudent
        m = ck["model_obj"]
        assert type(m) is DistilSCDStudent, type(m)
        assert type(m).__module__ == "distilscd.model"
        assert ck["epoch"] == 3
        with torch.no_grad():
            out = m(io["x1"], io["x2"])
        def same(a, b):
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            return torch.equal(a, b)
        for k, v in out.items():
            assert same(v, io["ref_" + k]), k
        print("OK", sorted(out))
    """)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout + res.stderr


def test_handwritten_protocol0_pickle():
    """Pickle protocole 0 écrit à la main : GLOBAL 'csf_mamba.losses.composite CSFMambaLoss'."""
    blob = b"ccsf_mamba.losses.composite\nCSFMambaLoss\np0\n."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cls = pickle.loads(blob)
    from distilscd.losses.composite import DistilSCDLoss
    assert cls is DistilSCDLoss


def test_state_dict_checkpoint_unaffected(tmp_path):
    """Les checkpoints réels (state_dict nu / dict de reprise) ne contiennent
    aucune référence de module : ils se chargent en weights_only=True."""
    from distilscd.model import DistilSCDStudent

    model = DistilSCDStudent(num_semantic_classes=7, encoder="conv", backend="ref")
    torch.save(model.state_dict(), tmp_path / "best.pt")
    torch.save({"model": model.state_dict(), "epoch": 1, "best_sek": 0.1, "global_step": 10},
               tmp_path / "last.pt")
    for name in ("best.pt", "last.pt"):
        raw = (tmp_path / name).read_bytes()
        assert b"csf_mamba" not in raw and b"distilscd" not in raw
        state = torch.load(tmp_path / name, map_location="cpu", weights_only=True)
        state = state["model"] if "model" in state else state
        DistilSCDStudent(num_semantic_classes=7, encoder="conv", backend="ref").load_state_dict(state)


if __name__ == "__main__":
    import tempfile

    test_shim_aliases_same_modules()
    test_shim_emits_deprecation_warning()
    with tempfile.TemporaryDirectory() as d:
        test_unpickle_old_module_path_after_rename(Path(d))
    test_handwritten_protocol0_pickle()
    with tempfile.TemporaryDirectory() as d:
        test_state_dict_checkpoint_unaffected(Path(d))
    print("OK")
