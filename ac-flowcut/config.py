"""Configuration for the released FlowCut/AC-FlowCut inference patch."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Union


@dataclass
class TokenReductionConfig:
    method: Literal["none", "flowcut"] = "flowcut"
    target_num: int = 128
    vision_stage_enabled: bool = True
    llm_stage_enabled: Optional[bool] = None
    vision_target_factor_anyres: float = 0.5
    llm_prune_layer: int = 1
    vision_select_layer_offset: int = 2
    prune_interval: int = 2
    ema_alpha: float = 0.5
    vision_prune_schedule_mode: Literal["instant_entropy", "ema_entropy"] = "instant_entropy"
    vision_score_normalization_mode: Literal["raw", "sum1"] = "raw"
    vision_score_accumulation_mode: Literal["ema"] = "ema"
    trace_enabled: bool = False
    trace_dir: Optional[str] = None
    strict_global_visual_budget: bool = True
    vision_selection_mode: Literal["topk", "attention_coreset"] = "topk"
    attention_coreset_protect_core: bool = True
    attention_coreset_query_scope: Literal["all", "cls"] = "all"
    llm_selection_mode: Literal["flowcut"] = "flowcut"

    def __post_init__(self):
        if self.method not in {"none", "flowcut"}:
            raise ValueError(f"Unknown token reduction method: {self.method}")
        if self.target_num < 1:
            raise ValueError("target_num must be >= 1")
        if self.prune_interval < 1:
            raise ValueError("prune_interval must be >= 1")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0, 1]")
        if self.vision_prune_schedule_mode not in {"instant_entropy", "ema_entropy"}:
            raise ValueError(f"Unknown vision_prune_schedule_mode: {self.vision_prune_schedule_mode}")
        if self.vision_score_normalization_mode not in {"raw", "sum1"}:
            raise ValueError(f"Unknown vision_score_normalization_mode: {self.vision_score_normalization_mode}")
        if self.vision_score_accumulation_mode != "ema":
            raise ValueError("AC-FlowCut release supports only vision_score_accumulation_mode='ema'")
        if self.vision_selection_mode not in {"topk", "attention_coreset"}:
            raise ValueError(f"Unknown vision_selection_mode: {self.vision_selection_mode}")
        if self.attention_coreset_query_scope not in {"all", "cls"}:
            raise ValueError(f"Unknown attention_coreset_query_scope: {self.attention_coreset_query_scope}")
        if self.llm_selection_mode != "flowcut":
            raise ValueError(
                "LLM attention-output repair is not supported. "
                "Use llm_selection_mode='flowcut'; AC repair runs only in the vision encoder."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_config(
    config_or_method: Optional[Union[str, Dict[str, Any], TokenReductionConfig]] = None,
    **kwargs,
) -> TokenReductionConfig:
    if isinstance(config_or_method, TokenReductionConfig):
        data = config_or_method.to_dict()
        data.update(kwargs)
    elif isinstance(config_or_method, dict):
        data = dict(config_or_method)
        data.update(kwargs)
    else:
        data = dict(kwargs)
        if config_or_method is not None:
            data["method"] = config_or_method

    # Accept redundant old JSON settings, but never let a second budget
    # override target_num. The removed option is not serialized again.
    legacy_target = data.pop("vision_target_num", None)
    target_num = data.get("target_num", TokenReductionConfig.target_num)
    if legacy_target is not None and legacy_target != target_num:
        raise ValueError(
            "vision_target_num has been removed; target_num is the only token budget. "
            "Delete vision_target_num from the configuration."
        )

    method = data.get("method", "flowcut")
    if method == "ac_flowcut":
        data["method"] = "flowcut"
        data.setdefault("vision_selection_mode", "attention_coreset")
    return TokenReductionConfig(**data)
