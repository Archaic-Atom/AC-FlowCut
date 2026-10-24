"""Runtime utilities used by FlowCut and AC-FlowCut.

This module intentionally contains only scoring, budget, and attention-output
selection helpers required by the released inference path.  Experiment
analysis and alternative selector implementations are kept outside this
package.
"""

import torch


CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*, defaults `False`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*, defaults `True`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def compute_score(cls_attn, semantic_attn, value):
    """Compute FlowCut's relation plus semantic/value evidence score."""
    relation_score = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    semantic_score = semantic_attn / semantic_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    value_magnitude = torch.norm(value, p=1, dim=-1)
    return (relation_score + semantic_score) * value_magnitude


def normalized_attention_entropy_ratio(attn):
    """Return attention entropy normalized by the maximum possible entropy."""
    n = attn.shape[-1]
    probability = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    entropy = -torch.sum(probability * torch.log(probability.clamp_min(1e-6)), dim=-1)
    maximum = torch.log(torch.tensor(float(max(n, 1)), device=attn.device, dtype=attn.dtype))
    return entropy / maximum.clamp_min(1e-6)


def adaptive_prune_ratio_from_entropy(entropy_ratio, n, remaining_layers, target_num):
    ratio = (n - target_num) / max(float(remaining_layers), 1.0) ** 0.5
    ratio = ratio * (1.0 - entropy_ratio**2)
    if ratio.shape[0] > 1:
        ratio = ratio.max(0).values
    return int(ratio.round())


def adaptive_prune_ratio(attn, remaining_layers, target_num, entropy_ratio=None):
    if entropy_ratio is None:
        entropy_ratio = normalized_attention_entropy_ratio(attn)
    return adaptive_prune_ratio_from_entropy(
        entropy_ratio=entropy_ratio,
        n=attn.shape[-1],
        remaining_layers=remaining_layers,
        target_num=target_num,
    )


def safe_normalize(x, dim=-1, eps=1e-6):
    return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)


def attention_entropy(attn, dim=-1, eps=1e-6):
    probability = safe_normalize(attn.clamp_min(0), dim=dim, eps=eps)
    return -(probability * torch.log(probability.clamp_min(eps))).sum(dim=dim)


def normalized_distribution_information(score, dim=-1, eps=1e-6):
    """Return parameter-free concentration in [0, 1]."""
    probability = safe_normalize(score.clamp_min(0), dim=dim, eps=eps)
    entropy = -(probability * torch.log(probability.clamp_min(eps))).sum(dim=dim)
    cardinality = score.shape[dim]
    if cardinality <= 1:
        return torch.ones_like(entropy)
    maximum = torch.log(torch.tensor(float(cardinality), device=score.device, dtype=probability.dtype))
    return (1.0 - entropy / maximum.clamp_min(eps)).clamp(0.0, 1.0)


def topk_keep(score, target_num, dim=-1, preserve_order=True):
    target_num = min(int(target_num), score.shape[dim])
    keep = torch.topk(score, target_num, dim=dim).indices
    return keep.sort(dim=dim).values if preserve_order else keep


def attention_output_coreset_select(
    flow_score,
    attn_weights,
    value_metric,
    target_num,
    protect_core=True,
    query_scope="all",
    eps=1e-6,
):
    """Greedily select a boundary set that preserves normalized attention output.

    ``flow_score`` supplies the FlowCut ranking and the protected prefix.  The
    remaining budget-derived boundary positions are chosen by minimizing the
    mean squared difference between the complete attention output and the
    output after renormalizing over CLS plus the selected patch keys.
    """
    if flow_score.ndim != 1:
        raise ValueError("flow_score must be one-dimensional")
    if attn_weights is None or value_metric is None:
        return topk_keep(flow_score.unsqueeze(0), int(target_num), dim=-1).squeeze(0)
    if attn_weights.ndim != 3 or value_metric.ndim != 2:
        raise ValueError("attention/value tensors must be [heads,tokens,tokens] and [tokens,dim]")
    heads, sequence, key_sequence = attn_weights.shape
    if sequence != key_sequence or value_metric.shape[0] != sequence or sequence != flow_score.numel() + 1:
        return topk_keep(flow_score.unsqueeze(0), int(target_num), dim=-1).squeeze(0)
    if query_scope not in {"all", "cls"}:
        raise ValueError(f"unknown query_scope: {query_scope}")

    n = int(flow_score.numel())
    k = min(int(target_num), n)
    baseline = topk_keep(flow_score.unsqueeze(0), k, dim=-1).squeeze(0)
    if k >= n or k <= 1:
        return baseline
    finite = torch.isfinite(flow_score).all() and torch.isfinite(attn_weights).all() and torch.isfinite(value_metric).all()
    if not finite:
        return baseline

    residual_slots = min(max(1, int(round(k**0.5))), k, n - k)
    ranked = torch.argsort(flow_score.float(), descending=True, stable=True)
    if protect_core:
        protected = ranked[: k - residual_slots]
        band = ranked[k - residual_slots : min(n, k + residual_slots)]
        selection_count = residual_slots
    else:
        protected = ranked[:0]
        band = ranked[: min(n, k + 2 * residual_slots)]
        selection_count = k
    if band.numel() < selection_count:
        return baseline

    attention = attn_weights.float().mean(dim=0)
    if query_scope == "cls":
        attention = attention[:1]
    values = value_metric.float()
    full_output = attention @ values
    selected_key_indices = torch.cat((torch.zeros(1, dtype=torch.long, device=flow_score.device), protected + 1))
    base_attention = attention[:, selected_key_indices]
    base_mass = base_attention.sum(dim=-1)
    base_sum = base_attention @ values[selected_key_indices]
    available = torch.ones(band.numel(), dtype=torch.bool, device=band.device)

    for _ in range(selection_count):
        positions = torch.nonzero(available, as_tuple=False).flatten()
        if positions.numel() == 0:
            break
        candidate_keys = band[positions] + 1
        candidate_mass = attention[:, candidate_keys]
        candidate_values = values[candidate_keys]
        candidate_output = (
            base_sum.unsqueeze(1) + candidate_mass.unsqueeze(-1) * candidate_values.unsqueeze(0)
        ) / (base_mass.unsqueeze(1) + candidate_mass).unsqueeze(-1).clamp_min(eps)
        errors = ((candidate_output - full_output.unsqueeze(1)) ** 2).mean(dim=(0, 2))
        best = torch.argmin(errors)
        chosen_position = positions[best]
        chosen_key = band[chosen_position] + 1
        chosen_mass = attention[:, chosen_key]
        base_sum = base_sum + chosen_mass.unsqueeze(-1) * values[chosen_key].unsqueeze(0)
        base_mass = base_mass + chosen_mass
        protected = torch.cat((protected, band[chosen_position : chosen_position + 1]))
        available[chosen_position] = False
    return protected.sort().values


def summarize_tensor(x):
    if x is None:
        return {}
    detached = x.detach().float()
    return {
        "mean": float(detached.mean().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }
