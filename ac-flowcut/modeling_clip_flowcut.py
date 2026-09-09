import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass,make_dataclass
from typing import Any, Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPConfig,CLIPVisionConfig
from transformers.utils import add_start_docstrings_to_model_forward,replace_return_docstrings
from functools import wraps
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from .utils_flowcut import CLIP_VISION_INPUTS_DOCSTRING


def CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()
    # get query proj
    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    value_metric = value_states.clone()

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) 

    cls_value = value_states[:,:1]
    semantic_weight = torch.bmm(cls_value, value_states.transpose(1,2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    # apply the causal_attention_mask first
    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                f" {causal_attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)


    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    semantic_attn = nn.functional.softmax(semantic_weight, dim=-1).view(bsz, self.num_heads, src_len)

    # FlowCut scores CLS-to-patch attention inside the encoder layer, even
    # when callers do not request public attention outputs.  Keep the shaped
    # tensor available for that internal path; the stock CLIP convention only
    # controls whether the encoder exposes it in its final ModelOutput.
    attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
    attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)
    
    return attn_output, attn_weights_reshaped, semantic_attn.mean(1), value_metric.mean(1)


def CLIPEncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
    target: int = 577,
    rpl: int = 22,
    cumulative_scores: torch.Tensor = None,
    cumulative_relation_scores: torch.Tensor = None,
    cumulative_semantic_scores: torch.Tensor = None,
    cumulative_entropy_ratio: torch.Tensor = None,
    persistence: torch.Tensor = None,
    cumulative_bridge_scores: torch.Tensor = None,
    all_keep_indices: torch.Tensor = None,
    layer_idx: int = -1,
) -> Tuple[torch.FloatTensor]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            `(config.encoder_attention_heads,)`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
    """
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights, semantic_attn, v_metric = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states
    
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states 

    # ==================================== Token Reduction ================================== #
    strategy = getattr(self, "token_reduction_strategy", None)
    if strategy is None:
        strategy = getattr(getattr(self, "encoder_owner", None), "token_reduction_strategy", None)
    if strategy is None:
        strategy = getattr(self, "flowcut_strategy", None)

    if strategy is not None and (
        not strategy.enabled_for_vision() or hidden_states.shape[1] > target
    ):

        cls_attention = attn_weights[:, :, 0, 1:].mean(dim=1)        # [B, N]

        # Always accumulate the model-internal visual evidence, even when the
        # vision stage is disabled.  This keeps full-resolution patch metadata
        # available for a later LLM-stage selector without performing an
        # irreversible 576 -> 128 reduction first.
        scores = strategy.vision_score(cls_attention, semantic_attn, v_metric, layer_idx=layer_idx)
        if all_keep_indices is None or all_keep_indices.shape != scores.shape:
            raise RuntimeError(
                "FlowCut patch provenance must stay aligned with patch scores: "
                f"indices={None if all_keep_indices is None else tuple(all_keep_indices.shape)} "
                f"scores={tuple(scores.shape)} at layer {layer_idx}"
            )
        score_for_cumulative = scores
        accumulation_mode = getattr(strategy.config, "vision_score_accumulation_mode", "ema")
        if getattr(strategy.config, "vision_score_normalization_mode", "raw") == "sum1":
            from .utils_flowcut import safe_normalize
            score_for_cumulative = safe_normalize(scores, dim=-1)
            if cumulative_scores is not None:
                cumulative_scores = safe_normalize(cumulative_scores, dim=-1)
        cumulative_scores = strategy.update_vision_score_cumulative(
            cumulative_scores, score_for_cumulative
        )
        from .utils_flowcut import normalized_attention_entropy_ratio
        current_entropy_ratio = normalized_attention_entropy_ratio(cls_attention)
        if getattr(strategy.config, "vision_prune_schedule_mode", "instant_entropy") == "ema_entropy":
            cumulative_entropy_ratio = strategy.update_cumulative(
                cumulative_entropy_ratio, current_entropy_ratio
            )
        selection_scores = cumulative_scores

        if strategy.enabled_for_vision() and strategy.should_prune_vision(rpl):

            before = hidden_states.shape[1]
            schedule_entropy_ratio = (
                cumulative_entropy_ratio
                if getattr(strategy.config, "vision_prune_schedule_mode", "instant_entropy") == "ema_entropy"
                else None
            )
            prune_num = strategy.adaptive_prune_num(
                cls_attention, rpl, target-1, entropy_ratio=schedule_entropy_ratio
            )
            target_num = max(hidden_states.shape[1]-prune_num, target)
            scores_before = selection_scores
            patch_indices_before = all_keep_indices
  
            # Keep one position for the CLS token.  The optional tie-break is
            # intentionally restricted to the final visual pruning event so
            # earlier FlowCut selection and the 576 -> 64 compute trajectory
            # remain unchanged.
            selection_mode = getattr(strategy.config, "vision_selection_mode", "topk")
            contextual_states = None
            contextual_summaries = None
            # AC-FlowCut is deliberately applied only at the final visual cut.
            if selection_mode == "attention_coreset" and target_num == target:
                from .utils_flowcut import attention_output_coreset_select
                keep_indices = torch.stack(
                    [
                        attention_output_coreset_select(
                            flow_score=selection_scores[batch_index],
                            attn_weights=attn_weights[batch_index],
                            value_metric=v_metric[batch_index],
                            target_num=target_num - 1,
                            protect_core=getattr(strategy.config, "attention_coreset_protect_core", True),
                            query_scope=getattr(strategy.config, "attention_coreset_query_scope", "all"),
                        )
                        for batch_index in range(hidden_states.shape[0])
                    ],
                    dim=0,
                )
            else:
                keep_indices = strategy.select_from_score(selection_scores, target_num-1)

            # Preserve compact diagnostics from the exact pre-pruning decision.
            # They make score-boundary and spatial-coverage hypotheses testable
            # without retaining full per-token tensors in every trace file.
            selection_summaries = []
            for batch_index in range(hidden_states.shape[0]):
                sample_scores = scores_before[batch_index].float()
                sample_keep = keep_indices[batch_index]
                selected_mask = torch.zeros(
                    sample_scores.shape[0], dtype=torch.bool, device=sample_scores.device
                )
                selected_mask[sample_keep] = True
                selected_scores = sample_scores[selected_mask]
                rejected_scores = sample_scores[~selected_mask]
                selected_patches = patch_indices_before[batch_index, sample_keep]
                grid_size = getattr(strategy.config, "vision_patch_grid_size", None) or max(
                    1, int(torch.ceil(torch.sqrt(patch_indices_before[batch_index].max().float() + 1)).item())
                )
                rows = torch.div(selected_patches.long(), grid_size, rounding_mode="floor").float()
                columns = (selected_patches.long() % grid_size).float()
                row_span = float((rows.max() - rows.min()).item() / max(grid_size - 1, 1))
                column_span = float((columns.max() - columns.min()).item() / max(grid_size - 1, 1))
                selection_summary = {
                    "selection_mode": getattr(strategy.config, "vision_selection_mode", "topk"),
                    "selected_score_min": float(selected_scores.min().item()),
                    "selected_score_mean": float(selected_scores.mean().item()),
                    "rejected_score_max": float(rejected_scores.max().item()) if rejected_scores.numel() else None,
                    "rejected_score_mean": float(rejected_scores.mean().item()) if rejected_scores.numel() else None,
                    "boundary_margin": (
                        float((selected_scores.min() - rejected_scores.max()).item())
                        if rejected_scores.numel() else None
                    ),
                    "selected_patch_count": int(selected_patches.numel()),
                    "selected_patch_unique": int(torch.unique(selected_patches).numel()),
                    "selected_row_span": row_span,
                    "selected_column_span": column_span,
                    "selected_patch_indices": selected_patches.detach().cpu().tolist(),
                    "patch_grid_size": int(grid_size),
                    "prune_schedule_mode": getattr(
                        strategy.config, "vision_prune_schedule_mode", "instant_entropy"
                    ),
                    "score_normalization_mode": getattr(
                        strategy.config, "vision_score_normalization_mode", "raw"
                    ),
                    "score_accumulation_mode": getattr(
                        strategy.config, "vision_score_accumulation_mode", "ema"
                    ),
                    "current_entropy_ratio": float(current_entropy_ratio[batch_index].item()),
                    "schedule_entropy_ratio": float(
                        (
                            schedule_entropy_ratio[batch_index]
                            if schedule_entropy_ratio is not None
                            else current_entropy_ratio[batch_index]
                        ).item()
                    ),
                    "bridge_selection_mode": selection_mode == "attention_coreset" and target_num == target,
                }
                trace_owner = getattr(self, "encoder_owner", self)
                trace_state = getattr(trace_owner, "token_reduction_trace", None)
                if (
                    selection_mode == "attention_coreset"
                    and target_num == target
                    and getattr(trace_state, "enabled", False)
                ):
                    from .utils_flowcut import (
                        attention_output_error_decomposition,
                        attention_output_reconstruction_error,
                    )
                    flowcut_keep = strategy.select_from_score(
                        scores_before[batch_index].unsqueeze(0), target_num - 1
                    ).squeeze(0)
                    coreset_protect_core = getattr(
                        strategy.config, "attention_coreset_protect_core", True
                    )
                    coreset_query_scope = getattr(
                        strategy.config, "attention_coreset_query_scope", "all"
                    )
                    selection_summary["attention_output_error"] = {
                        "flowcut_topk": float(
                            attention_output_reconstruction_error(
                                attn_weights[batch_index],
                                v_metric[batch_index],
                                flowcut_keep,
                            ).item()
                        ),
                        "coreset": float(
                            attention_output_reconstruction_error(
                                attn_weights[batch_index],
                                v_metric[batch_index],
                                sample_keep,
                                # Error is independent of the selector's
                                # protection/query options; the selected set
                                # The selected set already encodes the
                                # configured protection/query policy.
                            ).item()
                        ),
                        "protect_core": coreset_protect_core,
                        "query_scope": coreset_query_scope,
                    }
                    selection_summary["attention_output_error_decomposition"] = {
                        "flowcut_topk": attention_output_error_decomposition(
                            attn_weights[batch_index],
                            v_metric[batch_index],
                            flowcut_keep,
                        ),
                        "coreset": attention_output_error_decomposition(
                            attn_weights[batch_index],
                            v_metric[batch_index],
                            sample_keep,
                        ),
                    }
                # Candidate-design instrumentation only: retain the existing
                # per-head CLS-to-patch evidence at the final decision.  This
                # leaves FlowCut's head-mean score and all model states intact.
                if (
                    getattr(strategy.config, "trace_head_consensus", False)
                    and target_num == target
                ):
                    per_head_cls = attn_weights[batch_index, :, 0, 1:].detach().float()
                    selection_summary.update({
                        "head_consensus_trace": {
                            "head_count": int(per_head_cls.shape[0]),
                            "preprune_patch_indices": patch_indices_before[batch_index].detach().cpu().tolist(),
                            "cumulative_flowcut_scores": sample_scores.detach().cpu().tolist(),
                            "final_keep_local_indices": sample_keep.detach().cpu().tolist(),
                            "per_head_cls_attention": per_head_cls.transpose(0, 1).cpu().tolist(),
                        }
                    })
                selection_summaries.append(selection_summary)
            
            topk_states = contextual_states if contextual_states is not None else torch.gather(hidden_states[:,1:,:], dim=1, index=keep_indices.unsqueeze(-1).expand(-1,-1,hidden_states.size(-1)))
            cumulative_scores = torch.gather(cumulative_scores, dim=1, index=keep_indices)
            if accumulation_mode in {"channel_union_ema", "channel_geomean_ema"}:
                cumulative_relation_scores = torch.gather(
                    cumulative_relation_scores, dim=1, index=keep_indices
                )
                cumulative_semantic_scores = torch.gather(
                    cumulative_semantic_scores, dim=1, index=keep_indices
                )
            if persistence is not None:
                persistence = torch.gather(persistence, dim=1, index=keep_indices)
            if cumulative_bridge_scores is not None:
                cumulative_bridge_scores = torch.gather(
                    cumulative_bridge_scores, dim=1, index=keep_indices
                )
            cls_token = hidden_states[:,:1,:]

            all_keep_indices = torch.gather(all_keep_indices, dim=1, index=keep_indices)
            
            hidden_states = torch.cat([cls_token, topk_states], dim=1)
            strategy.trace_vision(
                owner=getattr(self, "encoder_owner", self),
                layer_idx=layer_idx,
                before=before,
                after=hidden_states.shape[1],
                target=target,
                prune_num=before-hidden_states.shape[1],
                cls_attention=cls_attention,
                keep_indices=keep_indices,
                scores=cumulative_scores,
                selection_summary=selection_summaries[0] if len(selection_summaries) == 1 else {
                    "batch_size": len(selection_summaries),
                    "samples": selection_summaries,
                },
            )

    # ==================================== Token Reduction ================================== #   

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return (
        outputs,
        cumulative_scores,
        cumulative_relation_scores,
        cumulative_semantic_scores,
        cumulative_entropy_ratio,
        persistence,
        cumulative_bridge_scores,
        all_keep_indices,
    )

def CLIPEncoder_forward(
    self,
    inputs_embeds,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutput]:

    # super().__init__()
    # self.config = config
    # self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    # self.gradient_checkpointing = False

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
   
    hidden_states = inputs_embeds
 
    # ``keep_indices`` produced below index ``hidden_states[:, 1:, :]`` and
    # therefore are patch-local (CLS is never a pruning candidate).  Keep the
    # provenance tensor in the same patch-only coordinate system from the very
    # first layer.  The previous implementation initialized it with
    # ``0..N`` including CLS and then gathered it with patch-local indices,
    # which shifted every spatial coordinate by one and made the metadata
    # length diverge from the score tensor after the first cut.
    patch_count = max(int(hidden_states.shape[1]) - 1, 0)
    all_keep_indices = torch.arange(patch_count, device=hidden_states.device).unsqueeze(0).expand(
        hidden_states.shape[0], -1
    )
    cumulative_scores = None
    cumulative_relation_scores = None
    cumulative_semantic_scores = None
    cumulative_entropy_ratio = None
    persistence = None
    cumulative_bridge_scores = None

    # Reset layer history for CW-FlowCut at the start of each forward pass
    if hasattr(self, "token_reduction_strategy") and hasattr(self.token_reduction_strategy, "reset_layer_history"):
        self.token_reduction_strategy.reset_layer_history()

    for idx, encoder_layer in enumerate(self.layers):
        encoder_layer.encoder_owner = self
        encoder_layer.token_reduction_strategy = getattr(self, "token_reduction_strategy", None)
        remain_prune_layer = self.select_layer-idx-1
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            (
                layer_outputs,
                cumulative_scores,
                cumulative_relation_scores,
                cumulative_semantic_scores,
                cumulative_entropy_ratio,
                persistence,
                cumulative_bridge_scores,
                all_keep_indices,
            ) = self._gradient_checkpointing_func(
                encoder_layer.__call__,
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions,
                self.target,
                remain_prune_layer,
                cumulative_scores,
                cumulative_relation_scores,
                cumulative_semantic_scores,
                cumulative_entropy_ratio,
                persistence,
                cumulative_bridge_scores,
                all_keep_indices,
                idx,
            )
        else:
            (
                layer_outputs,
                cumulative_scores,
                cumulative_relation_scores,
                cumulative_semantic_scores,
                cumulative_entropy_ratio,
                persistence,
                cumulative_bridge_scores,
                all_keep_indices,
            ) = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
                target = self.target,
                rpl = remain_prune_layer,
                cumulative_scores = cumulative_scores,
                cumulative_relation_scores=cumulative_relation_scores,
                cumulative_semantic_scores=cumulative_semantic_scores,
                cumulative_entropy_ratio = cumulative_entropy_ratio,
                persistence = persistence,
                cumulative_bridge_scores = cumulative_bridge_scores,
                all_keep_indices = all_keep_indices,
                layer_idx = idx,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    # Keep scores aligned with the surviving patch tokens for the optional
    # LLM-stage evidence-set selector.
    self.last_token_importance = cumulative_scores
    self.last_keep_indices = all_keep_indices

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None) + (all_keep_indices,)
    
    return BaseModelOutput(
        last_hidden_state=hidden_states, 
        hidden_states=encoder_states, 
        attentions=all_attentions,
    )


@add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
def CLIPVisionTransformer_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    hidden_states = self.embeddings(pixel_values, **kwargs)
    hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    all_keep_indices = self.encoder.last_keep_indices

    last_hidden_state = encoder_outputs[0]
    pooled_output = last_hidden_state[:, 0, :]
    pooled_output = self.post_layernorm(pooled_output)

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:] + (all_keep_indices, )

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def apply_flowcut_config(model, target_tokens):

    model.vision_model.encoder.target = target_tokens
    model.vision_model.encoder.select_layer = len(model.vision_model.encoder.layers) - 2


def apply_token_reduction_config(model, target_tokens, strategy, config=None, trace=None):
    encoder = model.vision_model.encoder
    encoder.target = target_tokens
    offset = getattr(config, "vision_select_layer_offset", 2) if config is not None else 2
    encoder.select_layer = len(encoder.layers) - offset
    encoder.token_reduction_strategy = strategy
    encoder.token_reduction_trace = trace
