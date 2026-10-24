from .clip_encoder import CLIPVisionTower_FlowCut
from .config import build_config
from .llava_arch import (
    encode_images_flowcut,
    encode_images_flowcut_multi,
    prepare_inputs_labels_for_multimodal_flowcut,
    restore_image_features_sorted,
)
from .llava_llama_flowcut import LlavaLlamaForCausalLM_forward, LlavaLlamaForCausalLM_generate
from .modeling_clip_flowcut import (
    CLIPAttention_forward,
    CLIPEncoder_forward,
    CLIPEncoderLayer_forward,
    CLIPVisionTransformer_forward,
    apply_token_reduction_config,
)
from .modeling_llama_flowcut import (
    LlamaDecoderLayer_forward,
    LlamaFlashAttention2_forward,
    LlamaModel_forward,
    LlamaSdpaAttention_forward,
)
from .registry import build_strategy
from .trace import ReductionTrace
from functools import wraps
import torch

# Import strategies for registration side effects.
from . import strategies  # noqa: F401


_LLAVA_PATCHED = False


def _refresh_accelerate_vision_forwards(vision_root):
    """Point existing Accelerate wrappers at the patched class forwards.

    ``CLIPVisionModel.from_pretrained(..., device_map="auto")`` may install
    an instance-level Accelerate wrapper before FlowCut patches the CLIP classes.
    That wrapper closes over the pre-patch method, so custom FlowCut state such
    as ``target`` never reaches ``CLIPEncoderLayer_forward``.  Rebuild only the
    wrapper closure while retaining its existing device/offload hook; removing
    and reattaching hooks here could disturb the loaded device map.
    """
    for module in vision_root.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        class_forward = getattr(type(module), "forward", None)
        if class_forward is None:
            continue
        patched_forward = class_forward.__get__(module, type(module))
        module._old_forward = patched_forward

        @wraps(patched_forward)
        def wrapped_forward(*args, _module=module, _forward=patched_forward, **kwargs):
            args, kwargs = _module._hf_hook.pre_forward(_module, *args, **kwargs)
            if _module._hf_hook.no_grad:
                with torch.no_grad():
                    output = _forward(*args, **kwargs)
            else:
                output = _forward(*args, **kwargs)
            return _module._hf_hook.post_forward(_module, output)

        module.forward = wrapped_forward


def _compute_vision_target(model, config):
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", "pad")
    select_feature = model.model.vision_tower.select_feature
    requested_target = config.target_num
    if image_aspect_ratio == "anyres":
        base = max(1, int(requested_target * config.vision_target_factor_anyres))
        return base + 1 if select_feature == "patch" else base
    return requested_target + 1 if select_feature == "patch" else requested_target


def _patch_llava_classes(patch_llm=True):
    global _LLAVA_PATCHED
    if not _LLAVA_PATCHED:
        from transformers.models.clip.modeling_clip import (
            CLIPAttention,
            CLIPEncoder,
            CLIPEncoderLayer,
            CLIPVisionTransformer,
        )

        CLIPAttention.forward = CLIPAttention_forward
        CLIPEncoderLayer.forward = CLIPEncoderLayer_forward
        CLIPEncoder.forward = CLIPEncoder_forward
        CLIPVisionTransformer.forward = CLIPVisionTransformer_forward

        from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

        CLIPVisionTower.forward = CLIPVisionTower_FlowCut.forward

        from llava.model.llava_arch import LlavaMetaForCausalLM

        if hasattr(LlavaMetaForCausalLM, "prepare_inputs_labels_for_multimodal"):
            LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
                prepare_inputs_labels_for_multimodal_flowcut
            )
            LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
            LlavaMetaForCausalLM.encode_images_flowcut_multi = encode_images_flowcut_multi
            LlavaMetaForCausalLM.encode_images_flowcut = encode_images_flowcut
        _LLAVA_PATCHED = True

    if patch_llm:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaFlashAttention2,
            LlamaModel,
            LlamaSdpaAttention,
        )

        LlamaModel.forward = LlamaModel_forward
        LlamaDecoderLayer.forward = LlamaDecoderLayer_forward
        LlamaSdpaAttention.forward = LlamaSdpaAttention_forward
        LlamaFlashAttention2.forward = LlamaFlashAttention2_forward

        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

        LlavaLlamaForCausalLM.forward = LlavaLlamaForCausalLM_forward
        LlavaLlamaForCausalLM.generate = LlavaLlamaForCausalLM_generate


def apply_token_reduction(model, config_or_method=None, **kwargs):
    config = build_config(config_or_method, **kwargs)
    strategy = build_strategy(config)
    trace = ReductionTrace(enabled=config.trace_enabled)

    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", "pad")
    if config.llm_stage_enabled is None:
        config.llm_stage_enabled = image_aspect_ratio == "anyres"

    if config.method != "none" and image_aspect_ratio == "anyres" and not config.llm_stage_enabled:
        raise ValueError(
            "Any-resolution pruning requires llm_stage_enabled=True for the final global "
            "target_num cut. This stage uses FlowCut top-k, without AC repair."
        )

    if config.method == "none":
        model.model.token_reduction_config = config
        model.model.token_reduction_strategy = strategy
        model.model.token_reduction_trace = trace
        return model

    _patch_llava_classes(patch_llm=config.llm_stage_enabled)

    # The vision tower is loaded before this function is called.  Its
    # Accelerate wrappers must therefore be refreshed after the class-level
    # FlowCut monkey patches are installed.
    # Refresh from the LLaVA tower wrapper downward.  Both this wrapper and
    # the underlying CLIP modules can have instance-level Accelerate forwards
    # captured before the class-level FlowCut patches are installed.
    _refresh_accelerate_vision_forwards(model.model.vision_tower)

    target_tokens = _compute_vision_target(model, config)
    apply_token_reduction_config(
        model.model.vision_tower.vision_tower,
        target_tokens=target_tokens,
        strategy=strategy,
        config=config,
        trace=trace,
    )

    model.model.token_reduction_config = config
    model.model.token_reduction_strategy = strategy
    model.model.token_reduction_trace = trace
    model.model.target_num = config.target_num
    model.model.prune_layer = config.llm_prune_layer

    return model


def flowcut(model, target_num=128, **kwargs):
    kwargs.setdefault("method", "flowcut")
    kwargs.setdefault("target_num", target_num)
    return apply_token_reduction(model, kwargs)


def ac_flowcut(model, target_num=128, **kwargs):
    """Apply the training-free Attention-Coreset FlowCut selector.

    FlowCut's progressive trajectory is retained.  At the final visual cut,
    AC-FlowCut protects the ranked core and greedily repairs a small boundary
    using the normalized multi-query attention-output proxy. Any-resolution
    models use ordinary FlowCut top-k at the LLM pruning layer to enforce the
    final global target_num budget; AC repair runs only in the vision encoder.
    """
    kwargs.setdefault("method", "flowcut")
    kwargs.setdefault("target_num", target_num)
    kwargs.setdefault("vision_selection_mode", "attention_coreset")
    return apply_token_reduction(model, kwargs)


def disable_token_reduction(model):
    return apply_token_reduction(model, method="none")
