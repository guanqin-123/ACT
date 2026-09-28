"""Checkpoint-backed BUFFET BERT inference from embedding sums."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertLayer


logger = logging.getLogger(__name__)


class NoVarLayerNorm(nn.Module):
    """BUFFET's affine mean-centering layer normalization without variance."""

    variant: str = "no_var"

    def __init__(self, hidden_size: int) -> None:
        """Initialize unit scale and zero bias."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mean-center the final dimension, then apply scale and bias."""
        centered = inputs - inputs.mean(dim=-1, keepdim=True)
        return self.weight * centered + self.bias


class BuffetEmbeddings(nn.Module):
    """The embedding LayerNorm portion of BUFFET's from-embeddings graph."""

    def __init__(self, hidden_size: int) -> None:
        """Create the no-variance embedding LayerNorm."""
        super().__init__()
        self.LayerNorm = NoVarLayerNorm(hidden_size)

    def forward(self, embedding_sum: torch.Tensor) -> torch.Tensor:
        """Apply BUFFET's embedding LayerNorm to a precomputed embedding sum."""
        return self.LayerNorm(embedding_sum)


class BuffetEncoder(nn.Module):
    """BERT encoder with the naming expected by ACT's route-A lowering."""

    def __init__(self, config: BertConfig) -> None:
        """Build eager-attention layers and install BUFFET LayerNorm variants."""
        super().__init__()
        self.layer = nn.ModuleList()
        for layer_index in range(config.num_hidden_layers):
            block = BertLayer(config, layer_idx=layer_index)
            setattr(
                block.attention.output,
                "LayerNorm",
                NoVarLayerNorm(config.hidden_size),
            )
            setattr(block.output, "LayerNorm", NoVarLayerNorm(config.hidden_size))
            self.layer.append(block)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run every encoder block without an attention mask."""
        for block in self.layer:
            output = block(hidden_states, attention_mask=None)
            hidden_states = output if isinstance(output, torch.Tensor) else output[0]
        return hidden_states


class BuffetPooler(nn.Module):
    """Dense/tanh pooler over the final hidden state of ``[CLS]``."""

    def __init__(self, hidden_size: int) -> None:
        """Create the pooler projection and tanh activation."""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool position zero."""
        return self.activation(self.dense(hidden_states[:, 0]))


class BuffetFromEmbeddings(nn.Module):
    """BUFFET classifier whose public input is the raw three-table embedding sum."""

    def __init__(self, config: BertConfig, num_labels: int = 2) -> None:
        """Build embedding LN, encoder, pooler, and classifier modules."""
        super().__init__()
        self._act_conversion_route = "b"
        self.embeddings = BuffetEmbeddings(config.hidden_size)
        self.encoder = BuffetEncoder(config)
        self.pooler = BuffetPooler(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, embedding_sum: torch.Tensor) -> torch.Tensor:
        """Classify a float tensor of shape ``[B, L, hidden_size]``."""
        dimension = embedding_sum.dim()
        if isinstance(dimension, int) and dimension != 3:
            raise ValueError("embedding_sum must have shape [B, L, hidden_size]")
        hidden_states = self.embeddings(embedding_sum)
        hidden_states = self.encoder(hidden_states)
        return self.classifier(self.pooler(hidden_states))


class BuffetRouteBNoVarNorm(nn.Module):
    """Pure-tensor form of BUFFET's affine mean-centering normalization."""

    def __init__(self, source: Any) -> None:
        super().__init__()
        self.scale = nn.Parameter(source.weight.detach().clone())
        self.shift = nn.Parameter(source.bias.detach().clone())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        centered = value - value.mean(dim=-1, keepdim=True)
        return self.scale * centered + self.shift


class BuffetRouteBBlock(nn.Module):
    """One BUFFET encoder block expressed as traceable tensor operations."""

    def __init__(self, source: Any, sequence_length: int) -> None:
        super().__init__()
        attention = source.attention
        self.q_proj = copy.deepcopy(attention.self.query)
        self.k_proj = copy.deepcopy(attention.self.key)
        self.v_proj = copy.deepcopy(attention.self.value)
        self.attn_proj = copy.deepcopy(attention.output.dense)
        self.attn_norm = BuffetRouteBNoVarNorm(attention.output.LayerNorm)
        self.ff_in = copy.deepcopy(source.intermediate.dense)
        self.relu = nn.ReLU()
        self.ff_out = copy.deepcopy(source.output.dense)
        self.ff_norm = BuffetRouteBNoVarNorm(source.output.LayerNorm)
        self.softmax = nn.Softmax(dim=-1)
        self.sequence_length = int(sequence_length)
        self.num_heads = int(attention.self.num_attention_heads)
        self.head_dim = int(attention.self.attention_head_size)
        self.hidden_size = self.num_heads * self.head_dim

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(
            -1, self.sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.q_proj(hidden))
        key = self._heads(self.k_proj(hidden))
        value = self._heads(self.v_proj(hidden))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probabilities = self.softmax(scores)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).contiguous().reshape(
            -1, self.sequence_length, self.hidden_size
        )
        attention_output = self.attn_norm(self.attn_proj(context) + hidden)
        intermediate = self.relu(self.ff_in(attention_output))
        return self.ff_norm(self.ff_out(intermediate) + attention_output)


class BuffetRouteBModel(nn.Module):
    """Traceable whole-tensor BUFFET model used by generic FX route B."""

    def __init__(self, source: BuffetFromEmbeddings, sequence_length: int) -> None:
        super().__init__()
        self.input_norm = BuffetRouteBNoVarNorm(source.embeddings.LayerNorm)
        self.stages = nn.ModuleList(
            BuffetRouteBBlock(block, sequence_length) for block in source.encoder.layer
        )
        self.pool_proj = copy.deepcopy(source.pooler.dense)
        self.pool_activation = nn.Tanh()
        self.output_proj = copy.deepcopy(source.classifier)

    def forward(self, embedding_sum: torch.Tensor) -> torch.Tensor:
        hidden = self.input_norm(embedding_sum)
        for stage in self.stages:
            hidden = stage(hidden)
        pooled = self.pool_activation(self.pool_proj(hidden[:, 0]))
        return self.output_proj(pooled)


def build_buffet_route_b_model(
    source: BuffetFromEmbeddings, sequence_length: int
) -> BuffetRouteBModel:
    """Create the generic-FX tensor graph for one verification sequence length."""
    return BuffetRouteBModel(source, sequence_length).train(source.training)


@dataclass(frozen=True)
class BuffetTokenizedInput:
    """WordPiece tokens and integer inputs derived from ACT loader tokens."""

    wordpiece_tokens: list[str]
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor


@dataclass(frozen=True)
class LoadedBuffetModel:
    """A strict BUFFET model load plus the tables needed to form its input."""

    model: BuffetFromEmbeddings
    tokenizer: BertTokenizer
    word_embeddings: torch.Tensor
    position_embeddings: torch.Tensor
    token_type_embeddings: torch.Tensor
    model_dir: Path
    checkpoint_dir: Path
    config: dict[str, object]
    unused_checkpoint_keys: tuple[str, ...]


def resolve_checkpoint(model_dir: str | Path) -> Path:
    """Resolve ``ckpt-N`` from a BUFFET model directory's checkpoint file."""
    directory = Path(model_dir).resolve()
    checkpoint_value = (directory / "checkpoint").read_text(encoding="utf-8").strip()
    if not checkpoint_value.isdigit():
        raise ValueError(f"{directory}/checkpoint does not contain a numeric checkpoint")
    checkpoint_dir = directory / f"ckpt-{int(checkpoint_value)}"
    for filename in ("config.json", "pytorch_model.bin", "vocab.txt"):
        if not (checkpoint_dir / filename).is_file():
            raise FileNotFoundError(checkpoint_dir / filename)
    return checkpoint_dir


def _bert_config(values: Mapping[str, object]) -> BertConfig:
    """Create the eager-attention HF configuration matching a BUFFET config."""
    config = BertConfig.from_dict(dict(values))
    config.layer_norm_eps = 1e-12
    config._attn_implementation = "eager"
    return config


def _parameter_key_map(model: BuffetFromEmbeddings) -> dict[str, str]:
    """Map every from-embeddings parameter explicitly to its checkpoint key."""
    mapping: dict[str, str] = {}
    for target_key in model.state_dict():
        if target_key.startswith(("embeddings.", "encoder.", "pooler.")):
            source_key = f"bert.{target_key}"
        elif target_key.startswith("classifier."):
            source_key = target_key
        else:
            raise RuntimeError(f"no BUFFET checkpoint mapping for {target_key}")
        mapping[target_key] = source_key
    return mapping


def load_buffet_model(
    model_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    strict: bool = True,
) -> LoadedBuffetModel:
    """Load a BUFFET checkpoint with explicit mapping and unused-key reporting."""
    directory = Path(model_dir).resolve()
    checkpoint_dir = resolve_checkpoint(directory)
    config_values: dict[str, object] = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    raw_state = torch.load(
        checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    if not isinstance(raw_state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in raw_state.items()
    ):
        raise TypeError("BUFFET checkpoint must be a string-to-tensor mapping")

    checkpoint_state = dict(cast(Mapping[str, torch.Tensor], raw_state))
    model = BuffetFromEmbeddings(_bert_config(config_values))
    key_map = _parameter_key_map(model)
    missing_sources = sorted(set(key_map.values()) - set(checkpoint_state))
    if missing_sources:
        raise RuntimeError(f"checkpoint is missing mapped keys: {missing_sources}")
    mapped_state = {
        target_key: checkpoint_state[source_key]
        for target_key, source_key in key_map.items()
    }
    model.load_state_dict(mapped_state, strict=True)

    embedding_keys = {
        "word": "bert.embeddings.word_embeddings.weight",
        "position": "bert.embeddings.position_embeddings.weight",
        "token_type": "bert.embeddings.token_type_embeddings.weight",
    }
    missing_tables = sorted(set(embedding_keys.values()) - set(checkpoint_state))
    if missing_tables:
        raise RuntimeError(f"checkpoint is missing embedding tables: {missing_tables}")
    consumed = set(key_map.values()) | set(embedding_keys.values())
    unused = tuple(sorted(set(checkpoint_state) - consumed))
    logger.info("Unused BUFFET checkpoint keys for %s: %s", directory, list(unused))
    if strict and unused:
        raise RuntimeError(f"unused checkpoint keys: {list(unused)}")

    model.to(device=device, dtype=dtype).eval()
    tokenizer = BertTokenizer(
        vocab_file=str(checkpoint_dir / "vocab.txt"), do_lower_case=True
    )
    return LoadedBuffetModel(
        model=model,
        tokenizer=tokenizer,
        word_embeddings=checkpoint_state[embedding_keys["word"]].to(
            device=device, dtype=dtype
        ),
        position_embeddings=checkpoint_state[embedding_keys["position"]].to(
            device=device, dtype=dtype
        ),
        token_type_embeddings=checkpoint_state[embedding_keys["token_type"]].to(
            device=device, dtype=dtype
        ),
        model_dir=directory,
        checkpoint_dir=checkpoint_dir,
        config=config_values,
        unused_checkpoint_keys=unused,
    )


def tokenize_act_tokens(
    tokens: Sequence[str], tokenizer: BertTokenizer
) -> BuffetTokenizedInput:
    """Reproduce BUFFET WordPiece conversion starting from ACT loader tokens."""
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    if not isinstance(cls_token, str) or not isinstance(sep_token, str):
        raise ValueError("checkpoint tokenizer does not define [CLS] and [SEP]")
    wordpieces: list[str] = [cls_token]
    wordpieces.extend(tokenizer.tokenize(" ".join(tokens)))
    wordpieces.append(sep_token)
    converted = tokenizer.convert_tokens_to_ids(wordpieces)
    if not isinstance(converted, list) or not all(isinstance(item, int) for item in converted):
        raise TypeError("tokenizer did not return a list of integer input IDs")
    input_ids = torch.tensor(converted, dtype=torch.long)
    return BuffetTokenizedInput(
        wordpiece_tokens=wordpieces,
        input_ids=input_ids,
        token_type_ids=torch.zeros_like(input_ids),
    )


def embedding_sum(
    loaded: LoadedBuffetModel,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum checkpoint word, position, and token-type tables without LayerNorm."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [L] or [B, L]")
    input_ids = input_ids.to(device=loaded.word_embeddings.device)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    elif token_type_ids.dim() == 1:
        token_type_ids = token_type_ids.unsqueeze(0)
    token_type_ids = token_type_ids.to(device=loaded.word_embeddings.device)
    if token_type_ids.shape != input_ids.shape:
        raise ValueError("token_type_ids must have the same shape as input_ids")

    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    positions = positions.unsqueeze(0).expand_as(input_ids)
    return (
        F.embedding(input_ids, loaded.word_embeddings)
        + F.embedding(positions, loaded.position_embeddings)
        + F.embedding(token_type_ids, loaded.token_type_embeddings)
    )
