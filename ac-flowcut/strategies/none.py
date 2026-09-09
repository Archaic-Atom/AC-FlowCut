from ..registry import register_strategy
from .base import TokenReductionStrategy


@register_strategy("none")
class NoneStrategy(TokenReductionStrategy):
    def enabled_for_vision(self):
        return False

    def enabled_for_llm(self):
        return False

    def vision_score(self, cls_attention, semantic_attn, value_metric, **kwargs):
        return None

    def adaptive_prune_num(self, attn, rpl, target_num, entropy_ratio=None):
        return 0

    def llm_score_and_select(self, **kwargs):
        return None
