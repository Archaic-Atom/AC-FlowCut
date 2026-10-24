import torch

from ..registry import register_strategy
from ..utils_flowcut import adaptive_prune_ratio, compute_score, summarize_tensor
from .base import TokenReductionStrategy


@register_strategy("flowcut")
class FlowCutStrategy(TokenReductionStrategy):
    def vision_score(self, cls_attention, semantic_attn, value_metric, **kwargs):
        return compute_score(cls_attention, semantic_attn[:, 1:], value_metric[:, 1:])

    def adaptive_prune_num(self, attn, rpl, target_num, entropy_ratio=None):
        return (
            adaptive_prune_ratio(attn, rpl, target_num, entropy_ratio=entropy_ratio)
            if rpl else attn.shape[-1]
        )

    def llm_score_and_select(
        self,
        query_states,
        key_states,
        value_states,
        hidden_states,
        visual_token_pos,
        visual_token_num,
        last_instruct_pos,
        target_num,
        head_dim,
        layer_idx,
        owner=None,
        text_token_pos=None,
        text_token_num=None,
        visual_token_indices=None,
        visual_token_importance=None,
        visual_token_grid_size=None,
    ):
        keep_indices = []
        merged_states = []
        bsz = query_states.shape[0]
        for b in range(bsz):
            if visual_token_num[b] <= target_num:
                keep_indices.append(None)
                merged_states.append(None)
                continue
            start = visual_token_pos[b]
            end = start + visual_token_num[b]
            vis_k = key_states[b, :, start:end]
            vis_v = value_states[b, :, start:end]
            text_q = query_states[b, :, last_instruct_pos[b] - 1:last_instruct_pos[b]]
            text_v = value_states[b, :, last_instruct_pos[b] - 1:last_instruct_pos[b]]
            relation_weight = torch.bmm(text_q, vis_k.transpose(1, 2)) / (head_dim ** 0.5)
            relation_attn = torch.nn.functional.softmax(
                relation_weight, dim=-1, dtype=torch.float32
            ).to(query_states.dtype).mean(0)
            semantic_weight = torch.bmm(text_v, vis_v.transpose(1, 2)) / (head_dim ** 0.5)
            semantic_attn = torch.nn.functional.softmax(
                semantic_weight, dim=-1, dtype=torch.float32
            ).to(query_states.dtype).mean(0)
            score = compute_score(relation_attn, semantic_attn, vis_v.mean(0)).squeeze(0)
            # AnyRes still needs a final global cut. Use the original FlowCut
            # ranking here; attention-output repair is confined to the ViT.
            keep = self.select_from_score(score.unsqueeze(0), target_num).squeeze(0)
            keep_indices.append(keep)
            merged_states.append(None)
            if owner is not None:
                self.trace_llm(
                    owner=owner,
                    sample_id=b,
                    layer_idx=layer_idx,
                    before=visual_token_num[b],
                    after=target_num,
                    target=target_num,
                    prune_num=visual_token_num[b] - target_num,
                    keep_indices=keep,
                    score_summary={
                        "final": summarize_tensor(score),
                        "selection_mode": "flowcut",
                    },
                )
        return {"keep_indices": keep_indices, "merged_states": merged_states}
