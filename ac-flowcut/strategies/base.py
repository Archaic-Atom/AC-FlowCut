from typing import Any, Dict, Optional

import torch

from ..trace import ReductionTraceEvent
from ..utils_flowcut import attention_entropy, summarize_tensor, topk_keep


class TokenReductionStrategy:
    def __init__(self, config):
        self.config = config

    @property
    def method(self):
        return self.config.method

    def enabled_for_vision(self):
        return self.config.method != "none" and self.config.vision_stage_enabled

    def enabled_for_llm(self):
        return self.config.method != "none" and self.config.llm_stage_enabled

    def update_cumulative(self, cumulative_scores, current_scores):
        if cumulative_scores is None:
            return current_scores.clone()
        alpha = self.config.ema_alpha
        return (1.0 - alpha) * cumulative_scores + alpha * current_scores

    def update_vision_score_cumulative(self, cumulative_scores, current_scores):
        if cumulative_scores is None:
            return current_scores.clone()
        return self.update_cumulative(cumulative_scores, current_scores)

    def should_prune_vision(self, rpl):
        return rpl % self.config.prune_interval == 0

    def vision_score(self, cls_attention, semantic_attn, value_metric, **kwargs):
        raise NotImplementedError

    def adaptive_prune_num(self, attn, rpl, target_num, entropy_ratio=None):
        raise NotImplementedError

    def select_from_score(self, score, target_num):
        return topk_keep(score, target_num, dim=-1, preserve_order=True)

    def llm_score_and_select(self, **kwargs):
        raise NotImplementedError

    def trace_vision(
        self,
        owner,
        layer_idx,
        before,
        after,
        target,
        prune_num,
        cls_attention,
        keep_indices,
        scores,
        selection_summary=None,
    ):
        trace = getattr(owner, "token_reduction_trace", None)
        if trace is None:
            return
        entropy = attention_entropy(cls_attention).detach().float().mean()
        trace.add(
            ReductionTraceEvent(
                stage="vision",
                layer_idx=layer_idx,
                num_tokens_before=before,
                num_tokens_after=after,
                target_num=target,
                prune_num=prune_num,
                entropy=float(entropy.item()),
                keep_indices=keep_indices[0].detach().cpu().tolist() if keep_indices is not None else None,
                score_summary={
                    "final": summarize_tensor(scores),
                    "selection": selection_summary or {},
                },
            )
        )

    def trace_llm(self, owner, sample_id, layer_idx, before, after, target, prune_num, keep_indices, score_summary):
        trace = getattr(owner, "token_reduction_trace", None)
        if trace is None:
            return
        trace.add(
            ReductionTraceEvent(
                stage="llm",
                layer_idx=layer_idx,
                sample_id=sample_id,
                num_tokens_before=before,
                num_tokens_after=after,
                target_num=target,
                prune_num=prune_num,
                keep_indices=keep_indices.detach().cpu().tolist() if keep_indices is not None else None,
                score_summary=score_summary,
            )
        )
