"""Loss Lovász-Softmax — portage de Mamba-FCS (Berman et al., 2018).

Source : third_party/MambaFCS/changedetection/utils_func/lovasz_loss.py

**Pourquoi cette loss ici.** Le diagnostic des matrices de confusion montre que
le goulot d'étranglement est la LOCALISATION du changement (ratio erreurs de
localisation / erreurs sémantiques : 3,5 sur Hi-UCD, 5,2 sur SECOND), avec un IoU
du changement de 39,7 % et 54,6 % respectivement. Or la Lovász-Softmax est une
extension convexe de l'IoU : elle **optimise directement** cette quantité, là où
la cross-entropy n'optimise que la justesse pixel par pixel. Et le SeK dépend de
l'IoU du changement via son facteur `exp(IoU_fg)`.

Deux écarts (sûrs) par rapport au fichier original :
  - `classes is 'present'` -> `classes == 'present'` : comparer des chaînes avec
    `is` déclenche une SyntaxWarning en Python moderne. Comportement identique.
  - suppression des `Variable(...)`, sans effet depuis PyTorch 0.4.
"""

import torch


def lovasz_grad(gt_sorted):
    """Gradient de l'extension de Lovász par rapport aux erreurs triées (Alg. 1)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cas d'un seul pixel
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _mean(values, empty=0.0):
    values = iter(values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        return empty
    for n, v in enumerate(values, 2):
        acc = acc + v
    return acc if n == 1 else acc / n


def flatten_probas(probas, labels, ignore=None):
    """Aplatit (B,C,H,W) -> (P,C) et (B,H,W) -> (P), en retirant les pixels ignorés."""
    if probas.dim() == 3:
        b, h, w = probas.size()
        probas = probas.view(b, 1, h, w)
    b, c, h, w = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def lovasz_softmax_flat(probas, labels, classes="present"):
    """probas: (P,C) probabilités ; labels: (P) entiers."""
    if probas.numel() == 0:
        return probas * 0.0
    c = probas.size(1)
    losses = []
    class_to_sum = list(range(c)) if classes in ("all", "present") else classes
    for cl in class_to_sum:
        fg = (labels == cl).float()
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[:, 0] if c == 1 else probas[:, cl]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm.data]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return _mean(losses)


def lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    """Lovász-Softmax multi-classe.

    probas : (B,C,H,W) probabilités (après softmax) ; labels : (B,H,W).
    ignore : indice de classe à ignorer (255 chez nous).
    """
    if per_image:
        return _mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
    return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
