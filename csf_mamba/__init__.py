"""Cale de compatibilité : `csf_mamba` est l'ancien nom du paquet `distilscd`.

Le projet a été renommé CSF-Mamba -> DistilSCD le 13 oct. 2026 (cf.
documentation/rename.md). Ce paquet ne contient aucun code propre : il
redirige tout import `csf_mamba.X` vers le module `distilscd.X` **lui-même**
(même objet dans `sys.modules`, pas une copie), de sorte que

* le vieux code (`from csf_mamba.model import CSFMamba`) continue de tourner ;
* `torch.load` / `pickle.load` d'un objet sérialisé sous `csf_mamba.*` le
  retrouve : pickle importe `csf_mamba.model`, obtient `distilscd.model`, puis
  y lit l'attribut — d'où les alias `CSFMamba` / `CSFMambaLoss` conservés dans
  `distilscd`.

La redirection est paresseuse (un *finder* placé en tête de `sys.meta_path`) :
on n'importe que les sous-modules réellement demandés, comme avant le renommage.
À retirer une fois tous les checkpoints et scripts migrés.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import distilscd as _distilscd

_OLD = __name__  # "csf_mamba"
_NEW = "distilscd"

warnings.warn(
    "le paquet 'csf_mamba' est renommé 'distilscd' ; importer 'distilscd' "
    "(cf. documentation/rename.md)",
    DeprecationWarning,
    stacklevel=2,
)


class _AliasLoader(importlib.abc.Loader):
    """Rend le module `distilscd.X` déjà importé au lieu d'en créer un nouveau."""

    def __init__(self, target: str):
        self.target = target
        self._real_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        # importlib écrase `__spec__` du module rendu ; on le restaure ensuite
        # pour que `distilscd.X` garde sa propre identité (reload, -m, etc.).
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self._real_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """`csf_mamba.X.Y` -> `distilscd.X.Y`, pour tout sous-module existant."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        new_name = _NEW + fullname[len(_OLD):]
        real = importlib.util.find_spec(new_name)
        if real is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(new_name),
            is_package=real.submodule_search_locations is not None,
        )


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

# Sous-modules de distilscd déjà chargés : alias immédiats.
for _name, _mod in list(sys.modules.items()):
    if _name.startswith(_NEW + "."):
        sys.modules.setdefault(_OLD + _name[len(_NEW):], _mod)


def __getattr__(name):
    # `csf_mamba.<attr>` -> `distilscd.<attr>` (attribut ou sous-module).
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(_distilscd, name)
    except AttributeError:
        pass
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
