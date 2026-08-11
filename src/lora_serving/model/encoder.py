"""Generic BERT-family encoder with batched multi-tenant LoRA.

Works for any model whose AutoModel loads a BertModel-compatible architecture
(BERT, RoBERTa, multilingual-E5, BGE, etc.). Architecture dimensions and LoRA
target modules are read from LoraServingConfig — nothing is hardcoded.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN

from lora_serving.config import LoraServingConfig
from lora_serving.ops.head import LRHeadOps
from lora_serving.ops.lora import LoraOps
from lora_serving.weights.batch import BatchedLRWeights, LayerwiseBatchedWeights


class AttentionWithLora(nn.Module):
    """Multi-head self-attention that applies per-sample LoRA deltas to target projections.

    The base Q/K/V projections are standard frozen linear layers. For each module
    in config.target_modules, the LoRA delta (B*A*x) is computed via LoraOps
    and added to the base projection output.
    """

    def __init__(self, config: LoraServingConfig, lora_ops: LoraOps):
        super().__init__()
        H = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.target_modules = config.target_modules
        self.lora_ops = lora_ops

        self.query = nn.Linear(H, H)
        self.key = nn.Linear(H, H)
        self.value = nn.Linear(H, H)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        lora_weights: LayerwiseBatchedWeights,
        apply_lora: bool = True,
    ) -> Tensor:
        B, S, H = hidden_states.shape

        projections = {
            "query": self.query(hidden_states),
            "key": self.key(hidden_states),
            "value": self.value(hidden_states),
        }

        # Here we are calculating LORA delta in the activation space, rather than param space
        # delta = B * (A * x)
        # Note: This dynamically applies LoRA to any combination of 'query', 'key', or 'value'
        # based on what is configured in `self.target_modules` (via LoraServingConfig).
        # While it can apply to all three, in practice (and following the original LoRA
        # paper by Hu et al.), only `query` and `value` are typically targeted to maximize
        # parameter efficiency without sacrificing performance.
        # This implementation matches PEFT's behavior of only targeting the requested modules.
        # `apply_lora=False` skips the entire delta path (cat/shrink/expand/add); used by the
        # forward-breakdown benchmark to isolate the wall-clock cost of the LoRA path.
        if apply_lora:
            for module in self.target_modules:
                if module not in projections:
                    continue
                a = torch.cat(lora_weights.a[module], dim=0)  # (B, H, R)
                b = torch.cat(lora_weights.b[module], dim=0)  # (B, R, H)
                # project down to lora_rank (B, S, R)
                self.lora_ops.shrink(hidden_states, a)
                # project back up to hidden_size (B, S, H)
                self.lora_ops.expand(b)
                projections[module] = projections[module] + self.lora_ops.output

        # Reshape for multi-head attention: (B, num_heads, S, head_dim)
        def reshape(t: Tensor) -> Tensor:
            return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(projections["query"])
        k = reshape(projections["key"])
        v = reshape(projections["value"])

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # attention_mask: (B, S) → additive mask (B, 1, 1, S)
            additive_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0
            scores = scores + additive_mask

        probs = torch.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        out = torch.matmul(probs, v)
        return out.transpose(1, 2).contiguous().view(B, S, H)


class EncoderLayerWithLora(nn.Module):
    """Single transformer encoder layer with LoRA-augmented self-attention."""

    def __init__(self, config: LoraServingConfig, lora_ops: LoraOps):
        super().__init__()
        H = config.hidden_size

        self.attention = nn.ModuleDict({
            "self": AttentionWithLora(config, lora_ops),
            "output": nn.ModuleDict({
                "dense": nn.Linear(H, H),
                "LayerNorm": nn.LayerNorm(H, eps=config.layer_norm_eps),
            }),
        })
        self.intermediate = nn.ModuleDict({
            "dense": nn.Linear(H, config.intermediate_size),
        })
        # FFN activation from the HF config (gelu for BERT/XLM-R, but e.g.
        # gelu_new or relu elsewhere) — parameterless, so it doesn't affect
        # the strict state-dict load.
        self.intermediate_act = ACT2FN[config.hidden_act]
        self.output = nn.ModuleDict({
            "dense": nn.Linear(config.intermediate_size, H),
            "LayerNorm": nn.LayerNorm(H, eps=config.layer_norm_eps),
        })

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        lora_weights: LayerwiseBatchedWeights,
        apply_lora: bool = True,
    ) -> Tensor:
        attn_out = self.attention["self"](hidden_states, attention_mask, lora_weights, apply_lora)
        attn_out = self.attention["output"]["dense"](attn_out)
        attn_out = self.attention["output"]["LayerNorm"](attn_out + hidden_states)

        mid = self.intermediate_act(self.intermediate["dense"](attn_out))
        out = self.output["dense"](mid)
        out = self.output["LayerNorm"](out + attn_out)
        return out


class EncoderWithLora(PreTrainedModel):
    """BERT-family encoder with batched multi-tenant LoRA and LR classification heads.

    Loads base weights from HuggingFace via AutoModel. At inference time, accepts
    a batch where each sample may have a different LoRA adapter and LR head.

    Example:
        config = LoraServingConfig(model_name="intfloat/multilingual-e5-small", ...)
        model = EncoderWithLora.from_pretrained_serving(config)
        model.eval()

        output = model(input_ids, attention_mask, lora_weights, lr_weights, output_lr)
    """

    def __init__(self, hf_config: PretrainedConfig, serving_config: LoraServingConfig):
        super().__init__(hf_config)
        H = serving_config.hidden_size
        lora_ops = LoraOps(serving_config)

        # Position-id scheme. BERT/DistilBERT index positions as arange(S). RoBERTa
        # family (incl. XLM-RoBERTa / bge-m3) and MPNet derive positions from the
        # non-pad mask, offset by pad_idx (first real token = pad_idx + 1).
        self.pad_idx = hf_config.pad_token_id if hf_config.pad_token_id is not None else 0
        self.offset_positions = hf_config.model_type in {
            "roberta", "xlm-roberta", "xlm-roberta-xl", "camembert", "mpnet",
        }

        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(hf_config.vocab_size, H, hf_config.pad_token_id),
            "position_embeddings": nn.Embedding(hf_config.max_position_embeddings, H),
            "token_type_embeddings": nn.Embedding(hf_config.type_vocab_size, H),
            "LayerNorm": nn.LayerNorm(H, eps=serving_config.layer_norm_eps),
        })

        self.encoder = nn.ModuleDict({
            "layer": nn.ModuleList([
                EncoderLayerWithLora(serving_config, lora_ops)
                for _ in range(serving_config.num_layers)
            ])
        })

        self.pooler = nn.ModuleDict({
            "dense": nn.Linear(H, H, bias=True),
        })
        
        # Hardcoded to Tanh because this matches the standard activation used by
        # BertPooler, RobertaPooler, and XLMRobertaPooler in Hugging Face.
        # This allows us to load base pooler weights directly for BERT, RoBERTa, and BGE models.
        self.pooler_act = nn.Tanh()
        self.post_init()

    @classmethod
    def from_pretrained_serving(cls, serving_config: LoraServingConfig) -> "EncoderWithLora":
        """Create the model and load base weights from HuggingFace."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(serving_config.model_name)
        model = cls(hf_config, serving_config)
        state_dict = AutoModel.from_pretrained(
            serving_config.model_name,
            torch_dtype=serving_config.dtype,
        ).state_dict()
        # Strict: every parameter of this reimplementation must come from the
        # checkpoint. Pooler-less checkpoints (ELECTRA discriminators) fail
        # here by design — accepting the gap would run CLS through a randomly
        # initialised pooler in forward(), which silently disagrees with
        # HFEncoderWithLora.encode_pooled's raw-CLS fallback for the same
        # checkpoint. Serve those with the hook wrapper (--engine hf).
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            hint = ""
            if any(k.startswith("pooler.") for k in missing):
                hint = (
                    " — the checkpoint ships no pooler; use the HF hook "
                    "wrapper (--engine hf / HFEncoderWithLora)"
                )
            raise RuntimeError(
                f"{serving_config.model_name} does not match the BERT-family "
                f"layout: missing={missing} unexpected={unexpected}{hint}"
            )
        return model.to(device=serving_config.device, dtype=serving_config.dtype)

    def position_ids(self, input_ids: Tensor) -> Tensor:
        """Position ids matching the base model's embedding scheme.

        arange for BERT/DistilBERT; mask-offset for RoBERTa/XLM-RoBERTa/MPNet
        (HF create_position_ids_from_input_ids: pad tokens map to pad_idx,
        the first real token to pad_idx + 1).
        """
        B, S = input_ids.shape
        if not self.offset_positions:
            return torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        mask = input_ids.ne(self.pad_idx).int()
        return (torch.cumsum(mask, dim=1) * mask).long() + self.pad_idx

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        lora_weights: list[LayerwiseBatchedWeights],
        lr_weights: BatchedLRWeights,
        output_lr: Tensor,
        token_type_ids: Tensor | None = None,
        apply_lora: bool = True,
    ) -> Tensor:
        """Run a full forward pass for a mixed-tenant batch.

        Args:
            input_ids:      (B, S)
            attention_mask: (B, S) or None
            lora_weights:   List of LayerwiseBatchedWeights, one per encoder layer
            lr_weights:     BatchedLRWeights with coef (B, max_labels, H) and intercept (B, max_labels)
            output_lr:      Pre-allocated output tensor (B, 1, max_labels)
            token_type_ids: (B, S) or None
            apply_lora:     If False, skip the LoRA delta path entirely (base-only
                            forward). Used by benchmarks to isolate LoRA wall-clock cost.

        Returns:
            output_lr: filled with per-sample logit scores (B, 1, max_labels)
        """
        B, S = input_ids.shape
        device = input_ids.device

        position_ids = self.position_ids(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, S, dtype=torch.long, device=device)

        x = (
            self.embeddings["word_embeddings"](input_ids)
            + self.embeddings["position_embeddings"](position_ids)
            + self.embeddings["token_type_embeddings"](token_type_ids)
        )
        x = self.embeddings["LayerNorm"](x)

        for layer, layer_lora in zip(self.encoder["layer"], lora_weights):
            x = layer(x, attention_mask, layer_lora, apply_lora)

        pooled = self.pooler_act(self.pooler["dense"](x[:, 0]))  # CLS token

        LRHeadOps.predict_proba(
            pooled.unsqueeze(1),
            lr_weights.coef,
            lr_weights.intercept,
            output_lr,
        )
        return output_lr
