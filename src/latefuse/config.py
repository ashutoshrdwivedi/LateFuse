from dataclasses import dataclass, field

import torch
from transformers import AutoConfig


@dataclass
class LoraServingConfig:
    """All configuration needed to run LoRA serving. Pass this to every component.

    Dimensions are derived automatically from the HuggingFace model config —
    nothing is hardcoded.

    Example:
        config = LoraServingConfig(
            model_name="intfloat/multilingual-e5-small",
            lora_rank=8,
            batch_size=32,
            max_seq_len=512,
            target_modules=["query", "value"],
            device=torch.device("cuda:0"),
            dtype=torch.float32,
        )
    """

    model_name: str
    lora_rank: int
    batch_size: int
    max_seq_len: int
    target_modules: list[str]  # projection names to apply LoRA to, e.g. ["query", "value"]
    device: torch.device
    dtype: torch.dtype

    # Derived from model config — do not set manually
    hidden_size: int = field(init=False)
    num_layers: int = field(init=False)
    num_heads: int = field(init=False)
    head_dim: int = field(init=False)
    intermediate_size: int = field(init=False)
    layer_norm_eps: float = field(init=False)
    hidden_act: str = field(init=False)

    def __post_init__(self):
        hf = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = hf.hidden_size
        self.num_layers = hf.num_hidden_layers
        self.num_heads = hf.num_attention_heads
        self.head_dim = hf.hidden_size // hf.num_attention_heads
        self.intermediate_size = hf.intermediate_size
        self.layer_norm_eps = hf.layer_norm_eps
        self.hidden_act = getattr(hf, "hidden_act", "gelu")
