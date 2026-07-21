"""Backend SSM interchangeable.

Point de dé-risquage central du projet : rien ici n'impose `mamba_ssm` à
l'import. Le fallback PyTorch pur permet de faire tourner le modèle complet sur
CPU (tests de formes, debug) sans chaîne CUDA. Sur le cluster, si le kernel est
disponible, on bascule dessus sans changer une ligne d'appel.

    backend="auto"    -> mamba_ssm si importable, sinon fallback
    backend="mamba"   -> impose le kernel (erreur claire s'il manque)
    backend="ref"     -> impose le fallback PyTorch pur

⚠️ Le fallback fait un scan séquentiel en Python : correct mais lent, en O(L)
itérations. Réservé aux petites longueurs (tests). Ne pas entraîner avec.
"""

import math

import torch
from torch import nn


def mamba_ssm_available() -> bool:
    try:
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    return True


class _RefS6(nn.Module):
    """Implémentation de référence d'un bloc S6 sélectif, PyTorch pur.

    Suit Gu & Dao (2024) : A, B, C, Δ dépendants de l'entrée, discrétisation ZOH.
    Sert d'oracle de correction et de chemin CPU ; pas de kernel fusionné.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=4, padding=3,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        nn.init.uniform_(self.dt_proj.bias, math.log(1e-3), math.log(1e-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = self.conv1d(x_in.transpose(1, 2))[..., :length].transpose(1, 2)
        x_in = nn.functional.silu(x_in)

        dt, b_mat, c_mat = torch.split(
            self.x_proj(x_in), [1, self.d_state, self.d_state], dim=-1
        )
        dt = nn.functional.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        a_mat = -torch.exp(self.A_log)                      # (d_inner, N)
        da = torch.exp(dt.unsqueeze(-1) * a_mat)            # (B, L, d_inner, N)
        db = dt.unsqueeze(-1) * b_mat.unsqueeze(2)          # (B, L, d_inner, N)

        state = torch.zeros(
            x.shape[0], self.d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        outputs = []
        for t in range(length):
            state = da[:, t] * state + db[:, t] * x_in[:, t].unsqueeze(-1)
            outputs.append(torch.einsum("bdn,bn->bd", state, c_mat[:, t]))

        y = torch.stack(outputs, dim=1) + self.D * x_in
        return self.out_proj(y * nn.functional.silu(z))


class SSMScan(nn.Module):
    """Scan SSM sur une séquence (B, L, C) -> (B, L, C)."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, backend: str = "auto"):
        super().__init__()
        if backend not in {"auto", "mamba", "ref"}:
            raise ValueError(f"backend inconnu : {backend!r}")

        use_kernel = backend == "mamba" or (backend == "auto" and mamba_ssm_available())
        if use_kernel:
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError(
                    "backend='mamba' demandé mais mamba_ssm est introuvable. "
                    "Sur Alliance Canada : vérifier `avail_wheels mamba_ssm causal_conv1d`, "
                    "ou utiliser backend='ref' (lent, CPU)."
                ) from exc
            self.scan = Mamba(d_model=d_model, d_state=d_state, expand=expand)
            self.backend = "mamba"
        else:
            self.scan = _RefS6(d_model, d_state=d_state, expand=expand)
            self.backend = "ref"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scan(x)
